"""Render judge-ensemble viz tab — categorize 4096 judged samples by 4-judge agreement
and show inline thumbnails + verdict chips per sample.

Output: appends one new cat3 tab "Judge Ensemble 5k v2" to grounding-viz/index.html.
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
from collections import defaultdict

from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_filter import draw_red_dots, extract_image_from_tar, GT_NORM_BY_DS
from build_cat3_dataset_panels import TAR_ROOTS, resolve_tar, extract_for_ds

OUT = '/weka/oe-training-default/zixianm/yinuoy/grounding_rm/eval/results/red_5k_v3'
FILTER_ROOT = '/weka/oe-training-default/oe-encoder/grounding_filter_molmo2'

# Map manifest dataset → filter output subdir + schema
DS_TO_FILTER = {
    'vg':         ('vg_filtered',         'legacy'),
    'openimages': ('openimages_filtered', 'legacy'),
    'pixmo':      ('pixmo_filtered',      'legacy'),
    'grit_v2':    ('grit_v2_filtered',    'new'),
    'cc3m':       ('cc3m_filtered',       'new'),
}


def build_filter_index(datasets_needed):
    """Build {(ds, key): [record, ...]} index. Multiple regions can share a key_in_tar
    (cc3m/cc12m generate r0/r1/r2... per image), so values are LISTS to be disambiguated
    by phrase later in lookup."""
    idx = defaultdict(list)
    for ds in datasets_needed:
        if ds not in DS_TO_FILTER: continue
        sub, schema = DS_TO_FILTER[ds]
        for jf in sorted(glob.glob(os.path.join(FILTER_ROOT, sub, '*.jsonl'))):
            try:
                for line in open(jf):
                    try: r = json.loads(line)
                    except: continue
                    if r.get('error'): continue
                    if schema == 'legacy':
                        key = r['key']
                        info = r.get('info') or {}
                        idx[(ds, key)].append({
                            'gt': r.get('gt') or [], 'pred': r.get('pred') or [],
                            'kind': r.get('kind'),
                            'precision': info.get('precision', 0.0),
                            'recall': info.get('recall', 0.0),
                            'pred_raw': r.get('pred_raw', ''),
                            'tar': r['shard'],
                            'key_in_tar': key,
                            'phrase': r.get('phrase', ''),
                        })
                    else:
                        # New schema: key in manifest is the "key_in_tar" stripped from full key
                        # Full key like 'coyo_0_snappy/00000.tar/000000008/c0' → strip first '.tar' prefix
                        full = r['key']
                        parts = full.split('/')
                        tar_path = key_in_tar = None
                        for i, p in enumerate(parts):
                            if p.endswith('.tar') and i+1 < len(parts):
                                tar_path = '/'.join(parts[:i+1])
                                key_in_tar = parts[i+1]
                                break
                        if not key_in_tar: key_in_tar = full
                        gt = r.get('gt_bboxes') if r.get('kind') == 'bbox' else r.get('gt_points')
                        idx[(ds, key_in_tar)].append({
                            'gt': gt or [], 'pred': r.get('pred_pts') or [],
                            'kind': r.get('kind'),
                            'precision': r.get('precision', 0.0),
                            'recall': r.get('recall', 0.0),
                            'pred_raw': r.get('pred_raw', ''),
                            'tar': tar_path or '',
                            'key_in_tar': key_in_tar,
                            'phrase': r.get('phrase', ''),
                        })
            except Exception:
                continue
    return idx


def fmt_coords(coords, n_cap=8):
    """Format list of bboxes or points compactly for display."""
    if not coords: return '[]'
    s = []
    for c in coords[:n_cap]:
        if len(c) == 4:  # bbox
            s.append(f'[{c[0]:.0f},{c[1]:.0f},{c[2]:.0f},{c[3]:.0f}]')
        elif len(c) == 2:  # point
            s.append(f'({c[0]:.0f},{c[1]:.0f})')
        else:
            s.append(str(c))
    more = f' …+{len(coords)-n_cap}' if len(coords) > n_cap else ''
    return ', '.join(s) + more


def load_jsonl(name):
    p = os.path.join(OUT, name)
    d = {}
    if not os.path.isfile(p): return d
    for line in open(p):
        try: r = json.loads(line)
        except: continue
        k = (r['dataset'], str(r['key']), int(r.get('pair_idx', 0)))
        d[k] = r
    return d


def thumb_b64(img_path, max_side=400):
    if not os.path.isfile(img_path): return None
    try:
        im = Image.open(img_path).convert('RGB')
        w, h = im.size
        if max(w, h) > max_side:
            s = max_side / max(w, h)
            im = im.resize((int(w * s), int(h * s)))
        buf = io.BytesIO(); im.save(buf, 'JPEG', quality=80)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception: return None


# Map manifest dataset → tar root key (vg/oi/pixmo are same; grit_v2 maps to grit)
DS_TO_TAR_DS = {'vg':'vg','openimages':'openimages','pixmo':'pixmo','grit_v2':'grit','cc3m':'cc3m'}


def render_with_gt(ds, rec, max_side=400):
    """Re-render image from tar with both red pred dots + green GT overlay. Returns b64 or None.
    rec = manifest row (has tar, key_in_tar, gt, pred, kind)."""
    if not rec.get('tar') or not rec.get('key_in_tar'):
        return None
    tar_ds = DS_TO_TAR_DS.get(ds, ds)
    try:
        tar_path = resolve_tar(tar_ds, rec['tar'])
        if not tar_path: return None
        img_bytes = extract_for_ds(tar_ds, tar_path, rec['key_in_tar'])
        if not img_bytes: return None
        kind = rec.get('kind', 'bbox')
        gt = rec.get('gt', [])
        gt_bboxes = gt if kind == 'bbox' else None
        gt_points = gt if kind == 'point' else None
        im = draw_red_dots(img_bytes, rec.get('pred', []),
                           gt_bboxes=gt_bboxes, gt_points=gt_points, max_side=max_side,
                           gt_norm=GT_NORM_BY_DS.get(ds, False))
        buf = io.BytesIO(); im.save(buf, 'JPEG', quality=80)
        return base64.b64encode(buf.getvalue()).decode()
    except Exception:
        return None


def chip(label, verdict):
    bg = {'yes': '#0FCB8C', 'no': '#F0529C', 'unclear': '#B11BE8', 'error': '#666'}.get(verdict, '#999')
    return (f'<span style="background:{bg};color:#FAF2E9;padding:2px 7px;border-radius:4px;'
            f'font-size:10px;font-weight:700;margin-right:4px">{html.escape(label)}: {html.escape(verdict)}</span>')


def render_card(rec_m, vg, vq, vi, vmm):
    """rec_m = manifest row (now self-contained: gt/pred/precision/recall bundled in)."""
    # Re-render image from tar with red pred + green GT overlay using per-ds coord system.
    b64 = render_with_gt(rec_m['dataset'], rec_m) or thumb_b64(rec_m['img_path'])
    if not b64: return None
    phrase = html.escape(rec_m['phrase'][:300])
    ds = html.escape(rec_m['dataset'])
    f1 = rec_m.get('f1', 0)
    n_pred = rec_m.get('n_pred', 0); n_gt = rec_m.get('n_gt', 0)
    kind = rec_m.get('kind', '?')
    pr = rec_m.get('precision', 0.0)
    rc = rec_m.get('recall', 0.0)
    gt_str = html.escape(fmt_coords(rec_m.get('gt', [])))
    pred_str = html.escape(fmt_coords(rec_m.get('pred', [])))
    pred_raw = html.escape((rec_m.get('pred_raw') or '')[:200])
    chips = ''
    if vg: chips += chip('GPT-5', vg['verdict'])
    if vq: chips += chip('Qwen', vq['verdict'])
    if vi: chips += chip('InternVL', vi['verdict'])
    if vmm: chips += chip('GLM', vmm['verdict'])
    reasons = []
    for jn, jr in [('GPT-5', vg), ('Qwen', vq), ('InternVL', vi), ('GLM', vmm)]:
        if jr and jr.get('reason'):
            rs = html.escape(jr['reason'][:160])
            reasons.append(f'<div style="font-size:10px;color:rgba(10,50,53,0.65);margin-top:2px"><b>{jn}:</b> {rs}</div>')
    reasons_html = ''.join(reasons[:4])
    ST = 'display:grid;grid-template-columns:400px 1fr;gap:16px;padding:12px;margin:8px 0;background:#fff;border:1px solid rgba(10,50,53,0.15);border-radius:6px'
    F1_CHIP = f'<span style="background:#F0529C;color:#FAF2E9;padding:2px 7px;border-radius:4px;font-size:10px;font-weight:700">F1={f1:.2f}</span>'
    PR_CHIP = (f'<span style="background:#105257;color:#FAF2E9;padding:2px 7px;border-radius:4px;font-size:10px;font-weight:700;margin-left:4px">P={pr:.2f}</span>'
               f'<span style="background:#105257;color:#FAF2E9;padding:2px 7px;border-radius:4px;font-size:10px;font-weight:700;margin-left:2px">R={rc:.2f}</span>')
    return (f'<div style="{ST}">'
            f'<div><img src="data:image/jpeg;base64,{b64}" style="width:100%;border-radius:4px"/></div>'
            '<div>'
            f'<div style="color:rgba(10,50,53,0.55);font-family:ui-monospace,Menlo,monospace;font-size:11px;margin-bottom:6px">{ds} / {html.escape(str(rec_m["key"]))[:80]} · kind={kind}</div>'
            f'<div style="color:#0A3235;font-size:13px;margin-bottom:6px"><b>{phrase}</b></div>'
            f'<div style="margin-bottom:6px">{F1_CHIP}{PR_CHIP}<span style="margin-left:6px;font-size:11px;color:rgba(10,50,53,0.6)">n_pred={n_pred} n_gt={n_gt}</span></div>'
            f'<div style="font-size:11px;color:#0A3235;font-family:ui-monospace,Menlo,monospace;margin-bottom:2px"><b style="color:#0FCB8C">GT</b>: {gt_str}</div>'
            f'<div style="font-size:11px;color:#0A3235;font-family:ui-monospace,Menlo,monospace;margin-bottom:2px"><b style="color:#F0529C">PRED</b>: {pred_str}</div>'
            + (f'<div style="font-size:10px;color:rgba(10,50,53,0.55);font-family:ui-monospace,Menlo,monospace;margin-bottom:6px">pred_raw: {pred_raw}</div>' if pred_raw else '')
            + f'<div style="margin-top:6px">{chips}</div>'
            f'<div style="margin-top:4px">{reasons_html}</div>'
            '</div>'
            '</div>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per_cat', type=int, default=20, help='samples per category')
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--snippets_dir', default='/weka/oe-training-default/zixianm/yinuoy/grounding-viz/.snippets')
    args = ap.parse_args()

    manifest = {}
    for line in open(os.path.join(OUT, 'manifest.jsonl')):
        r = json.loads(line)
        manifest[(r['dataset'], str(r['key']), int(r.get('pair_idx', 0)))] = r

    g = load_jsonl('judge_gpt5.jsonl')
    q = load_jsonl('judge_qwen3vl_32b_instruct.jsonl')
    i = load_jsonl('judge_internvl3_78b.jsonl')
    m = load_jsonl('judge_glm4v_108b.jsonl')

    # Categorize on 4-judge intersection using BOTH mask_verdict + molmo_verdict (new 2-stage prompt).
    base_keys = set(manifest) & set(g) & set(q) & set(i) & set(m)
    cats = defaultdict(list)  # cat -> list of keys
    for k in base_keys:
        mvs = [g[k].get('mask_verdict','?'), q[k].get('mask_verdict','?'),
               i[k].get('mask_verdict','?'), m[k].get('mask_verdict','?')]
        movs = [g[k].get('molmo_verdict','?'), q[k].get('molmo_verdict','?'),
                i[k].get('molmo_verdict','?'), m[k].get('molmo_verdict','?')]
        n_mw = sum(v == 'mask_wrong' for v in mvs)
        n_y = sum(v == 'yes' for v in movs)
        n_n = sum(v == 'no' for v in movs)
        if n_mw == 4: cat = 'all_mask_wrong'
        elif n_mw >= 3: cat = 'maj_mask_wrong'
        elif n_y == 4: cat = 'all_yes'
        elif n_n == 4: cat = 'all_no'
        elif n_y >= 3 and n_y > n_n: cat = 'maj_yes'
        elif n_n >= 3 and n_n > n_y: cat = 'maj_no'
        else: cat = 'tied'
        cats[cat].append(k)

    print('Category sizes:', {c: len(ks) for c, ks in cats.items()}, flush=True)
    rng = random.Random(args.seed)

    # 7-category order: drop → recover → ambiguous → hard. Manifest carries gt/pred directly.
    cat_order_keys = ['all_mask_wrong', 'maj_mask_wrong', 'all_yes', 'maj_yes', 'tied', 'maj_no', 'all_no']
    picks_by_cat = {c: (rng.sample(cats[c], min(args.per_cat, len(cats[c]))) if c in cats else []) for c in cat_order_keys}

    cat_order = [
        ('all_mask_wrong', '🚫 ALL agree mask_wrong (DROP — GT broken consensus)'),
        ('maj_mask_wrong', '⚠️ Majority mask_wrong (likely drop)'),
        ('all_yes', '✅ ALL agree molmo=yes (RECOVER — Molmo correct, GT was the problem)'),
        ('maj_yes', '✅ Majority molmo=yes (likely recover)'),
        ('tied', '❓ Tied / ambiguous'),
        ('maj_no', '❌ Majority molmo=no (likely hard example)'),
        ('all_no', '❌ ALL agree molmo=no (HARD EXAMPLE — Molmo wrong on real GT)'),
    ]
    sections = []
    for cat, label in cat_order:
        picks = picks_by_cat.get(cat, [])
        cards = []
        for k in picks:
            mrec = manifest[k]
            html_card = render_card(mrec, g.get(k), q.get(k), i.get(k), m.get(k))
            if html_card: cards.append(html_card)
        if not cards: continue
        sections.append(
            f'<div style="margin-top:24px">'
            f'<div style="font-weight:700;color:#105257;font-size:15px;border-bottom:2px solid #105257;padding-bottom:4px;margin-bottom:8px">{label} ({len(cats.get(cat,[]))} total · showing {len(cards)})</div>'
            + ''.join(cards) + '</div>'
        )

    # Per-dataset breakdown
    ds_counts = defaultdict(lambda: defaultdict(int))
    for cat, ks in cats.items():
        for k in ks:
            ds_counts[manifest[k]['dataset']][cat] += 1
    breakdown_rows = ['<tr><th>dataset</th><th>n</th><th>all_mask_wr</th><th>maj_mask_wr</th><th>all_yes</th><th>maj_yes</th><th>tied</th><th>maj_no</th><th>all_no</th></tr>']
    for ds, cc in sorted(ds_counts.items()):
        tot = sum(cc.values())
        cells = ''.join(f'<td style="text-align:right;padding:4px 8px">{cc.get(c,0)} ({100*cc.get(c,0)/tot:.0f}%)</td>' for c in ['all_mask_wrong','maj_mask_wrong','all_yes','maj_yes','tied','maj_no','all_no'])
        breakdown_rows.append(f'<tr><td style="padding:4px 8px"><b>{ds}</b></td><td style="text-align:right;padding:4px 8px">{tot}</td>{cells}</tr>')
    breakdown_html = '<table style="margin-top:8px;border-collapse:collapse;font-size:12px"><tbody>' + ''.join(breakdown_rows) + '</tbody></table>'

    n_g = len(g); n_q = len(q); n_i = len(i); n_m = len(m); n_base = len(base_keys); n_glm = len(base_keys & set(m))
    intro = (
        '<div class="dataset-intro">'
        '<div class="intro-title"><b>4-judge ensemble on red_5k_v2 (1500 grit_v2 + 1500 cc3m + 700 vg + 300 oi + 300 pixmo)</b></div>'
        f'<div class="intro-desc">Each judged sample shows the 4-judge verdict chips + brief reasons. Categorized by agreement: ALL-AGREE-YES = definite false-reject (Molmo correct, GT too restrictive); ALL-AGREE-NO = real hard example (Molmo wrong); tied = ambiguous.</div>'
        f'<div class="intro-meta">Judges done: GPT-5={n_g}/4300, Qwen3-VL-32B={n_q}/4300, InternVL3-78B={n_i}/4300, GLM-4.5V={n_m}/4300 · 3-judge intersection={n_base} · 4-judge intersection={n_glm}</div>'
        f'{breakdown_html}'
        '</div>'
    )
    panel = (f'<div id="p_3_judge_ensemble_5kv3" class="panel" data-cat="cat3">\n{intro}\n'
             f'{"".join(sections)}\n</div>\n')
    tab = '<button class="ds-tab" data-panel="p_3_judge_ensemble_5kv3">5k Judge Ensemble</button>'

    snip_dir = getattr(args, 'snippets_dir', None) or '/weka/oe-training-default/zixianm/yinuoy/grounding-viz/.snippets'
    os.makedirs(snip_dir, exist_ok=True)
    with open(os.path.join(snip_dir, 'cat3_judge_tab.html'), 'w') as f: f.write(tab)
    with open(os.path.join(snip_dir, 'cat3_judge_panel.html'), 'w') as f: f.write(panel)
    print(f'wrote {snip_dir}/cat3_judge_tab.html ({len(tab)} bytes)')
    print(f'wrote {snip_dir}/cat3_judge_panel.html ({len(panel):,} bytes)')


if __name__ == '__main__':
    main()
