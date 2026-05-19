"""Rebuild cat3 (filter) section of grounding-viz index.html.

One tab per dataset (vg / openimages / pixmo / grit / cc3m / cc12m / seeclick / rf100),
each showing N random F1<1 rejected samples with red-dot pred + green GT overlay.

Replaces the entire cat3_tabs + all p_3_* panels with the new layout. Datasets without
any output yet render an empty "no data" placeholder so the tab still appears.
"""
import argparse
import base64
import glob
import html
import io
import json
import os
import random
import re
import sys
import tarfile

from PIL import Image, ImageDraw

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_filter import draw_red_dots, extract_image_from_tar, GT_NORM_BY_DS  # reuse

FILTER_ROOT = '/weka/oe-training-default/oe-encoder/grounding_filter_molmo2'

# Each dataset: (label, output_subdir, tar_root, key_parser, schema)
# schema = 'legacy' (shard + key + gt + pred) or 'new' (key with tar prefix + gt_bboxes/pred_pts)
TAR_ROOTS = {
    'vg':         '/weka/oe-training-default/oe-encoder/vg',
    'openimages': '/weka/oe-training-default/oe-encoder/openimages',
    'pixmo':      '/weka/oe-training-default/oe-encoder/pixmo',
    'grit':       '/weka/oe-training-default/oe-encoder/grit',
    'cc3m':       '/weka/oe-training-default/oe-encoder/cc3m/data',
    'cc12m':      '/weka/oe-training-default/oe-encoder/cc12m/data',
    'seeclick':   '/weka/oe-training-default/webolmo/datasets/seeclick',
    'rf100':      '/weka/oe-training-default/oe-encoder/roboflow100_tars',
}

DATASETS = [
    # (label,      out_subdir,           schema,   pretty)
    ('vg',         'vg_filtered',        'legacy', 'Visual Genome'),
    ('openimages', 'openimages_filtered','legacy', 'OpenImages V7'),
    ('pixmo',      'pixmo_filtered',     'legacy', 'Pixmo Point'),
    ('grit',       'grit_v2_filtered',   'new',    'GRIT v2'),
    ('cc3m',       'cc3m_filtered',      'new',    'cc3m'),
    ('cc12m',      'cc12m_filtered',     'new',    'cc12m'),
    ('seeclick',   'seeclick_filtered',  'new',    'SeeClick'),
    ('rf100',      'rf100_filtered',     'new',    'RF100'),
    ('molmoweb_sg', '', 'raw_arrow', 'MolmoWeb-SyntheticGround'),
]

MOLMOWEB_SG_ROOT = '/weka/oe-training-default/zixianm/molmoweb-datasets/MolmoWeb-SyntheticGround'


def parse_new_key(key):
    """'coyo_0_snappy/00000.tar/000000008/c0' → ('coyo_0_snappy/00000.tar', '000000008').
    For seeclick (<png>::<phrase>) → ('SEECLICK', <png>); rf100 (<task>/<task>-NNNNNNNN) → ('RF100', key)."""
    if '::' in key:  # seeclick
        return 'SEECLICK', key.split('::')[0]
    parts = key.split('/')
    for i, p in enumerate(parts):
        if p.endswith('.tar'):
            return '/'.join(parts[:i+1]), parts[i+1] if i+1 < len(parts) else parts[-1]
    # rf100: <task>/<task>-NNNNNNNN
    if len(parts) == 2 and parts[0] in {''} or parts[1].startswith(parts[0] + '-'):
        return 'RF100', key
    return None, None


def resolve_tar(ds, shard_or_path):
    """For legacy: shard is bare like '00000.tar', search by name. For new: shard_or_path is full relative path.
    Special sentinels 'SEECLICK' and 'RF100' returned by parse_new_key are handled in extract_for_ds."""
    if shard_or_path in ('SEECLICK', 'RF100'):
        return shard_or_path
    if '/' in shard_or_path:
        direct = os.path.join(TAR_ROOTS[ds], shard_or_path)
        return direct if os.path.isfile(direct) else None
    direct = os.path.join(TAR_ROOTS[ds], shard_or_path)
    if os.path.isfile(direct):
        return direct
    m = glob.glob(os.path.join(TAR_ROOTS[ds], '**', shard_or_path), recursive=True)
    return m[0] if m else None


def extract_for_ds(ds, tar_path, key_in_tar):
    """Dispatch image-bytes loading: seeclick=loose png, rf100=scan task dir tars, else=tar extract."""
    if tar_path == 'SEECLICK':
        # seeclick stores loose images under seeclick_web_imgs/
        img_p = os.path.join(TAR_ROOTS['seeclick'], 'seeclick_web_imgs', key_in_tar)
        if os.path.isfile(img_p):
            return open(img_p, 'rb').read()
        return None
    if tar_path == 'RF100':
        # key_in_tar = <task>/<task>-NNNNNNNN; file = <task>-NNNNNNNN.jpg in <task>/*.tar
        task, sample = key_in_tar.split('/', 1)
        # Try tar shard inferred from sample id (sample//10000 → tar NNNNN), then fall back to scan
        try:
            n = int(sample.split('-')[-1])
            tar_id = n // 10000
            candidate = os.path.join(TAR_ROOTS['rf100'], task, f'{task}-{tar_id:05d}.tar')
            if os.path.isfile(candidate):
                b = extract_image_from_tar(candidate, sample)
                if b: return b
        except Exception:
            pass
        for tp in glob.glob(os.path.join(TAR_ROOTS['rf100'], task, '*.tar')):
            b = extract_image_from_tar(tp, sample)
            if b: return b
        return None
    return extract_image_from_tar(tar_path, key_in_tar)


def load_rejected(ds, out_subdir, schema, max_pool=2000, seed=0):
    """Return list of rejected (F1<1) records normalized to {tar, key_in_tar, phrase, gt, pred, kind, f1, ...}.

    Bijection contract: deduplicates on composite key (tar, key_in_tar, phrase). Drops literal
    duplicates that filter writes (pixmo's worker shards have ~4% overlap from preempt-resume).
    Pool is shuffled deterministically (seed) for unbiased subsampling.
    """
    pat = os.path.join(FILTER_ROOT, out_subdir, '*.jsonl')
    files = sorted(glob.glob(pat))
    pool = []
    seen = set()  # dedup on (tar, key_in_tar, phrase)
    n_dup = 0
    for jf in files:
        try:
            for line in open(jf):
                try: r = json.loads(line)
                except: continue
                if r.get('error'): continue
                if schema == 'legacy':
                    info = r.get('info') or {}
                    f1 = info.get('f1', 1.0)
                    if f1 >= 1.0 - 1e-9: continue
                    rec = {
                        'tar': r['shard'], 'key_in_tar': r['key'],
                        'phrase': r.get('phrase', ''),
                        'gt': r.get('gt') or [], 'pred': r.get('pred') or [],
                        'kind': r.get('kind', 'bbox'),
                        'f1': f1, 'precision': info.get('precision', 0.0), 'recall': info.get('recall', 0.0),
                        'n_pred': info.get('n_pred', len(r.get('pred') or [])),
                        'n_gt': info.get('n_gt', len(r.get('gt') or [])),
                        'pred_raw': r.get('pred_raw', ''),
                    }
                else:
                    f1 = r.get('f1', 1.0)
                    if f1 >= 1.0 - 1e-9: continue
                    if not r.get('pred_pts'): continue
                    tar, key_in_tar = parse_new_key(r['key'])
                    if not tar: continue
                    gt = r.get('gt_bboxes') if r.get('kind') == 'bbox' else r.get('gt_points')
                    rec = {
                        'tar': tar, 'key_in_tar': key_in_tar,
                        'phrase': r.get('phrase', ''),
                        'gt': gt or [], 'pred': r['pred_pts'],
                        'kind': r.get('kind', 'bbox'),
                        'f1': f1, 'precision': r.get('precision', 0.0), 'recall': r.get('recall', 0.0),
                        'n_pred': r.get('n_pred', len(r['pred_pts'])),
                        'n_gt': r.get('n_gt', len(gt or [])),
                        'pred_raw': r.get('pred_raw', ''),
                    }
                composite = (rec['tar'], rec['key_in_tar'], rec['phrase'])
                if composite in seen:
                    n_dup += 1; continue
                seen.add(composite)
                pool.append(rec)
                if len(pool) >= max_pool: break
        except Exception:
            continue
        if len(pool) >= max_pool: break
    if n_dup:
        print(f'  load_rejected[{ds}]: deduplicated {n_dup} literal duplicates', flush=True)
    rng = random.Random(seed)
    rng.shuffle(pool)
    return pool, len(files)


def render_thumb(img_bytes, pred_pts, gt, kind, max_side=400, gt_norm=False):
    gt_bboxes = gt if kind == 'bbox' else None
    gt_points = gt if kind == 'point' else None
    im = draw_red_dots(img_bytes, pred_pts, gt_bboxes=gt_bboxes, gt_points=gt_points,
                       max_side=max_side, gt_norm=gt_norm)
    buf = io.BytesIO(); im.save(buf, 'JPEG', quality=80)
    return base64.b64encode(buf.getvalue()).decode()


def render_molmoweb_sg_panel(ds, pretty, n_samples, seed):
    """Special raw-arrow panel for MolmoWeb-SyntheticGround: no Molmo pred, just show
    GT bbox (green rect) + GT click point (green dot in percent of image dims)."""
    import json as _json
    try:
        from datasets import load_from_disk
    except ImportError:
        empty = f'<div id="p_3_{ds}" class="panel" data-cat="cat3"><div class="dataset-intro"><div class="intro-title"><b>{pretty}</b></div><div class="intro-desc">(skipped: datasets lib unavailable)</div></div></div>\n'
        return empty, pretty
    rng = random.Random(seed)
    cards = []
    n_done = n_err = n_total = 0
    for sub in ('gpt', 'template'):
        sub_dir = os.path.join(MOLMOWEB_SG_ROOT, sub)
        if not os.path.isdir(sub_dir):
            continue
        try:
            arr = load_from_disk(sub_dir)
            idx_path = os.path.join(sub_dir, 'image_index.json')
            img_index = _json.load(open(idx_path))
        except Exception as e:
            print(f'  {ds}/{sub}: load failed: {e}', flush=True)
            continue
        n_total += len(arr)
        picks = rng.sample(range(len(arr)), min(n_samples // 2, len(arr)))
        for ridx in picks:
            try:
                rec = arr[ridx]
                img_p = img_index.get(str(ridx))
                if not img_p or not os.path.isfile(img_p):
                    n_err += 1; continue
                with open(img_p, 'rb') as f: img_bytes = f.read()
                W, H = rec['metadata']['image_w'], rec['metadata']['image_h']
                # Pick one message at random
                msgs = rec.get('messages') or []
                if not msgs: n_err += 1; continue
                msg = rng.choice(msgs)
                phrase = msg.get('question') or ''
                bbox = msg.get('bbox') or []  # pixel xyxy
                try:
                    action = _json.loads(msg.get('answer', '{}')).get('action', {})
                    pct_x, pct_y = action.get('x'), action.get('y')
                    px, py = (pct_x * W / 100.0, pct_y * H / 100.0) if pct_x is not None else (None, None)
                except Exception:
                    px = py = None
                gt_pts = [[px, py]] if px is not None else []
                gt_box = [bbox] if len(bbox) == 4 else []
                im = draw_red_dots(img_bytes, [], gt_bboxes=gt_box, gt_points=gt_pts,
                                   max_side=400, gt_norm=False)  # MolmoWeb GT pre-converted to pixel
                buf = io.BytesIO(); im.save(buf, 'JPEG', quality=80)
                b64 = base64.b64encode(buf.getvalue()).decode()
                phrase_s = html.escape(phrase[:300])
                website = html.escape(rec['metadata'].get('website', ''))
                ST_SAMPLE = 'display:grid;grid-template-columns:400px 1fr;gap:16px;padding:12px;margin:8px 0;background:#fff;border:1px solid rgba(10,50,53,0.15);border-radius:6px'
                ST_META = 'color:rgba(10,50,53,0.55);font-family:ui-monospace,Menlo,monospace;font-size:11px;margin-bottom:6px'
                ST_TITLE = 'font-weight:700;color:#105257;font-size:11px;margin-top:8px;text-transform:uppercase;letter-spacing:.04em'
                ST_TEXT = 'color:#0A3235;font-size:13px;margin-top:2px'
                cards.append(
                    f'<div class="sample" style="{ST_SAMPLE}">\n'
                    f'  <div class="img-wrap" style="width:400px"><img src="data:image/jpeg;base64,{b64}" style="width:100%;border-radius:4px"/></div>\n'
                    '  <div>\n'
                    f'    <div style="{ST_META}">{sub} · {website} · {W}×{H} · rec {ridx}</div>\n'
                    f'    <div style="{ST_TITLE}">phrase (question)</div>\n'
                    f'    <div style="{ST_TEXT}"><b>{phrase_s}</b></div>\n'
                    f'    <div style="{ST_TITLE}">overlay</div>\n'
                    f'    <div style="{ST_TEXT};font-size:11px"><span style="color:#0FCB8C;font-weight:600">●</span> GT click point &nbsp; <span style="color:#0FCB8C;font-weight:600">□</span> GT bbox</div>\n'
                    + (f'    <div style="{ST_TITLE}">action</div>\n    <div style="{ST_TEXT};font-family:ui-monospace,Menlo,monospace;font-size:11px">x={pct_x}%, y={pct_y}% &nbsp; bbox={bbox}</div>\n' if px is not None else '')
                    + '  </div>\n</div>\n'
                )
                n_done += 1
            except Exception:
                n_err += 1
    print(f'  {ds}: rendered={n_done} err={n_err} (pool {n_total:,} across gpt+template)', flush=True)
    body = (f'<div style="margin-top:8px">{"".join(cards)}</div>' if n_done
            else '<div style="padding:20px;color:rgba(10,50,53,0.5)">No samples rendered.</div>')
    intro = (
        '<div class="dataset-intro">'
        f'<div class="intro-title"><b>{pretty} — raw training samples (allenai HF)</b></div>'
        f'<div class="intro-desc">Green box = GT bounding box of the target UI element. Green dot = GT click point (percent of image dims). No Molmo prediction yet — this is the raw web-grounding training data.</div>'
        f'<div class="intro-meta">{n_total:,} total samples across gpt+template; sampled {n_done}</div>'
        '</div>'
    )
    return f'<div id="p_3_{ds}" class="panel" data-cat="cat3">\n{intro}\n{body}\n</div>\n', pretty


def render_panel(ds, out_subdir, schema, pretty, n_samples, seed):
    pool, n_files = load_rejected(ds, out_subdir, schema, max_pool=n_samples * 10, seed=seed)
    n_total = len(pool)
    picks = pool[:n_samples]
    cards = []
    n_done = n_err = 0
    for rec in picks:
        try:
            tar_path = resolve_tar(ds, rec['tar'])
            if not tar_path: n_err += 1; continue
            img_bytes = extract_for_ds(ds, tar_path, rec['key_in_tar'])
            if not img_bytes: n_err += 1; continue
            b64 = render_thumb(img_bytes, rec['pred'], rec['gt'], rec['kind'],
                               gt_norm=GT_NORM_BY_DS.get(ds, False))
            phrase = html.escape((rec['phrase'] or '')[:300])
            pred_raw = html.escape((rec.get('pred_raw') or '')[:300])
            ST_SAMPLE = 'display:grid;grid-template-columns:400px 1fr;gap:16px;padding:12px;margin:8px 0;background:#fff;border:1px solid rgba(10,50,53,0.15);border-radius:6px'
            ST_META = 'color:rgba(10,50,53,0.55);font-family:ui-monospace,Menlo,monospace;font-size:11px;margin-bottom:6px'
            ST_TITLE = 'font-weight:700;color:#105257;font-size:11px;margin-top:8px;text-transform:uppercase;letter-spacing:.04em'
            ST_TEXT = 'color:#0A3235;font-size:13px;margin-top:2px'
            cards.append(
                f'<div class="sample" style="{ST_SAMPLE}">\n'
                f'  <div class="img-wrap" style="width:400px"><img src="data:image/jpeg;base64,{b64}" style="width:100%;border-radius:4px"/></div>\n'
                '  <div>\n'
                f'    <div style="{ST_META}">{ds} / {html.escape(rec["key_in_tar"])}</div>\n'
                f'    <div style="{ST_TITLE}">phrase</div>\n'
                f'    <div style="{ST_TEXT}"><b>{phrase}</b></div>\n'
                f'    <div style="{ST_TITLE}">metrics</div>\n'
                f'    <div style="{ST_TEXT}"><span style="background:#F0529C;color:#FAF2E9;font-weight:700;padding:2px 8px;border-radius:4px;font-size:11px">F1={rec["f1"]:.2f}</span> &nbsp; P={rec["precision"]:.2f} &nbsp; R={rec["recall"]:.2f} &nbsp; n_pred={rec["n_pred"]} &nbsp; n_gt={rec["n_gt"]}</div>\n'
                f'    <div style="{ST_TITLE}">overlay</div>\n'
                f'    <div style="{ST_TEXT};font-size:11px"><span style="color:#0FCB8C;font-weight:600">●</span> GT &nbsp;<span style="color:#F0529C;font-weight:600">●</span> Molmo pred</div>\n'
                + (f'    <div style="{ST_TITLE}">pred_raw</div>\n    <div style="{ST_TEXT};font-family:ui-monospace,Menlo,monospace;font-size:11px">{pred_raw}</div>\n' if pred_raw else '')
                + '  </div>\n</div>\n'
            )
            n_done += 1
        except Exception:
            n_err += 1
    print(f'  {ds}: pool={n_total} from {n_files} files, rendered {n_done}, err {n_err}', flush=True)
    if n_files == 0:
        body = '<div style="padding:20px;color:rgba(10,50,53,0.5)">No filter output yet (job still pending or running with no shards completed).</div>'
    elif n_done == 0:
        body = f'<div style="padding:20px;color:rgba(10,50,53,0.5)">{n_total:,} rejected samples in {n_files} jsonl files; render failed (tar resolution).</div>'
    else:
        body = f'<div style="margin-top:8px">{"".join(cards)}</div>'
    intro = (
        '<div class="dataset-intro">'
        f'<div class="intro-title"><b>{pretty} — Molmo rejected samples (F1&lt;1)</b></div>'
        f'<div class="intro-desc">Red = Molmo predicted point. Green = ground-truth (bbox + center, or point). Random sample from rejected pool — these are records the filter discarded.</div>'
        f'<div class="intro-meta">{n_total:,} rejected (F1&lt;1) across {n_files} shard jsonls · showing {n_done} random</div>'
        '</div>'
    )
    return f'<div id="p_3_{ds}" class="panel" data-cat="cat3">\n{intro}\n{body}\n</div>\n', pretty


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_per_ds', type=int, default=50)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--snippets_dir', default='/weka/oe-training-default/zixianm/yinuoy/grounding-viz/.snippets',
                    help='write cat3_dataset_tabs.html + cat3_dataset_panels.html here')
    args = ap.parse_args()

    print(f'Rendering {args.n_per_ds} samples per dataset (×{len(DATASETS)} datasets)...', flush=True)
    panels = []
    tabs = []
    for i, (ds, sub, schema, pretty) in enumerate(DATASETS):
        if schema == 'raw_arrow':
            panel_html, label = render_molmoweb_sg_panel(ds, pretty, args.n_per_ds, args.seed)
        else:
            panel_html, label = render_panel(ds, sub, schema, pretty, args.n_per_ds, args.seed)
        panels.append(panel_html)
        active = ' active' if i == 0 else ''
        tabs.append(f'<button class="ds-tab{active}" data-panel="p_3_{ds}">{html.escape(label)}</button>')

    tabs_html = '<div id="cat3_tabs" class="ds-tabs">' + ''.join(tabs) + '</div>\n'
    panels_html = ''.join(panels)

    os.makedirs(args.snippets_dir, exist_ok=True)
    tabs_out = os.path.join(args.snippets_dir, 'cat3_dataset_tabs.html')
    panels_out = os.path.join(args.snippets_dir, 'cat3_dataset_panels.html')
    with open(tabs_out, 'w') as f: f.write(tabs_html)
    with open(panels_out, 'w') as f: f.write(panels_html)
    print(f'\nwrote {tabs_out} ({len(tabs_html):,} bytes, {len(DATASETS)} tabs)')
    print(f'wrote {panels_out} ({len(panels_html):,} bytes, {len(panels)} panels)')


if __name__ == '__main__':
    main()
