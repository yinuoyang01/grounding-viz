"""Render the FULL model-spectrum harvest pool (pointing + reasonseg + filter)
as a cat3 ("Grounding data filtering") panel — writes .snippets/cat3_judge_tab.html
+ cat3_judge_panel.html, which generate.py injects.

Per-bench GT rendering:
  - pointing : binary mask (translucent fill)
  - reasonseg: polygons (rasterised to mask, translucent fill)
  - filter   : bbox outlines / point markers (in dataset coord convention)

Dot color reflects CONSENSUS QUALITY (not raw GT label):
  green  = clean_pos (GT-hit AND near model consensus)
  red    = clean_neg (GT-miss AND far from consensus = real model error)
  gray   = ambiguous (e.g. GT-miss but near consensus -> GT too strict)
  (no_pred dots are skipped)
"""
import argparse
import base64
import io
import json
import os
import random
import sys
import html as _html

sys.path.insert(0, '/weka/oe-training-default/zixianm/yinuoy/grounding_rm/data')
from eval_pointing import LOADERS as PT_LOADERS, BR as PT_BR  # noqa: E402
from eval_reasonseg import load_reasonseg  # noqa: E402
from eval_filter_sample import sample_ds as flt_sample, gt_pixel as flt_gt_px, SUB as FLT_SUB  # noqa: E402
from build_cat3_dataset_panels import resolve_tar, extract_for_ds  # noqa: E402
from judge_inline import DS_TO_TAR_DS  # noqa: E402

POOL_PATH = '/weka/oe-training-default/oe-encoder/rm_pool_full.jsonl'
RS_ROOT = '/weka/oe-training-default/royg/grounding_data/reasonseg/val'
SNIP = '/weka/oe-training-default/zixianm/yinuoy/grounding-viz/.snippets'
SHORT = {'molmo2-4b': 'Molmo2-4B', 'molmo7b-d': 'Molmo-7B', 'molmoE-1b': 'MolmoE-1B',
         'qwen2.5vl-3b': 'Qwen2.5VL-3B', 'qwen2.5vl-7b': 'Qwen2.5VL-7B',
         'qwen2.5vl-32b': 'Qwen2.5VL-32B', 'internvl3-8b': 'InternVL3-8B'}


def _img_bytes(r):
    return open(r['img_path'], 'rb').read() if 'img_path' in r else r['img_bytes']


def render_card(g, indexes):
    import numpy as np
    from PIL import Image, ImageDraw
    bench = g['bench']
    pt_idx, rs_idx, fl_idx = indexes
    try:
        if bench == 'pointing':
            r = pt_idx.get(g['item'])
            if not r: return None
            im = Image.open(io.BytesIO(_img_bytes(r))).convert('RGB'); W, H = im.size
            mb = open(r['mask_path'], 'rb').read() if 'mask_path' in r else r['mask_bytes']
            mk = Image.open(io.BytesIO(mb)).convert('L')
            if mk.size != (W, H): mk = mk.resize((W, H))
            gt = ('mask', np.array(mk))
        elif bench == 'reasonseg':
            r = rs_idx.get(g['item'])
            if not r: return None
            im = Image.open(r['img_path']).convert('RGB'); W, H = im.size
            tmp = Image.new('L', (W, H), 0); td = ImageDraw.Draw(tmp)
            for poly in r['polys']:
                td.polygon([(int(p[0]), int(p[1])) for p in poly], fill=255)
            gt = ('mask', np.array(tmp))
        elif bench == 'filter':
            x = fl_idx.get(g['item'])
            if not x: return None
            ds, r = x
            tp = resolve_tar(DS_TO_TAR_DS.get(ds, ds), r['tar'])
            if not tp: return None
            ib = extract_for_ds(DS_TO_TAR_DS.get(ds, ds), tp, r['key_in_tar'])
            if not ib: return None
            im = Image.open(io.BytesIO(ib)).convert('RGB'); W, H = im.size
            gt = ('shapes', flt_gt_px(ds, r, W, H))
        else:
            return None
    except Exception:
        return None

    # 1. Build CLEAN image (just resized, no overlays)
    sc = 540 / max(W, H)
    nW, nH = max(1, int(W * sc)), max(1, int(H * sc))
    clean = im.resize((nW, nH))

    # 2. Build MARKED image: GT overlay (at full res) -> resize -> dots + model labels
    marked_full = im.copy()
    if gt[0] == 'mask':
        yel = np.zeros((H, W, 4), dtype=np.uint8)
        yel[gt[1] > 10] = (255, 210, 0, 130)
        marked_full = Image.alpha_composite(marked_full.convert('RGBA'),
                                            Image.fromarray(yel, 'RGBA')).convert('RGB')
    else:
        d0 = ImageDraw.Draw(marked_full)
        lw = max(3, W // 280)
        for typ, c in gt[1]:
            if typ == 'bbox':
                d0.rectangle([min(c[0], c[2]), min(c[1], c[3]),
                              max(c[0], c[2]), max(c[1], c[3])],
                             outline=(255, 210, 0), width=lw)
            else:
                rr = max(8, W // 120)
                d0.ellipse([c[0] - rr, c[1] - rr, c[0] + rr, c[1] + rr],
                           outline=(255, 210, 0), width=lw)
    marked = marked_full.resize((nW, nH))
    d = ImageDraw.Draw(marked)
    rad = 10
    QCOL = {'clean_pos': (0, 200, 80), 'clean_neg': (235, 40, 60),
            'ambiguous': (140, 140, 140)}
    for p in g['preds']:
        q = p.get('quality', 'ambiguous')
        if q == 'no_pred':
            continue
        col = QCOL.get(q, (140, 140, 140))
        tag = SHORT.get(p['model'], p['model']).replace('-', '')[:5]
        for (x, y) in (p.get('pred') or []):
            cx, cy = x * sc, y * sc
            # ring
            d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad],
                      outline=(255, 255, 255), width=3)
            d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=col, width=3)
            # small text label next to the dot (offset right-down)
            tx, ty = cx + rad + 3, cy - rad - 2
            # text bg for contrast
            bb = d.textbbox((tx, ty), tag)
            d.rectangle([bb[0] - 2, bb[1] - 1, bb[2] + 2, bb[3] + 1], fill=(255, 255, 255))
            d.text((tx, ty), tag, fill=col)

    def _b64(im):
        b = io.BytesIO(); im.save(b, 'JPEG', quality=83)
        return base64.b64encode(b.getvalue()).decode()
    bc, bm = _b64(clean), _b64(marked)

    chips = ''
    for p in g['preds']:
        q = p.get('quality', 'ambiguous')
        bg = {'clean_pos': '#0FCB8C', 'clean_neg': '#F0529C',
              'ambiguous': '#999', 'no_pred': '#bbb'}.get(q, '#999')
        tag = {'clean_pos': '✓', 'clean_neg': '✗', 'ambiguous': '?', 'no_pred': '—'}.get(q, '?')
        chips += (f'<span style="background:{bg};color:#FAF2E9;padding:2px 7px;border-radius:4px;'
                  f'font-size:11px;font-weight:700;margin:0 4px 4px 0;display:inline-block">'
                  f'{_html.escape(SHORT.get(p["model"], p["model"]))} {tag}</span>')
    ST = ('display:grid;grid-template-columns:780px 1fr;gap:16px;padding:12px;margin:8px 0;'
          'background:var(--bg,#FAF2E9);border:1px solid rgba(10,50,53,0.15);border-radius:8px')
    return (f'<div style="{ST}">'
            f'<div style="display:flex;gap:8px">'
            f'<img src="data:image/jpeg;base64,{bc}" style="width:50%;border-radius:4px"/>'
            f'<img src="data:image/jpeg;base64,{bm}" style="width:50%;border-radius:4px"/>'
            f'</div>'
            f'<div>'
            f'<div style="color:rgba(10,50,53,0.55);font-family:ui-monospace,Menlo,monospace;'
            f'font-size:11px;margin-bottom:6px">{_html.escape(bench)} / '
            f'{_html.escape(str(g.get("category","-")))}</div>'
            f'<div style="color:#0A3235;font-size:14px;font-weight:700;margin-bottom:10px;line-height:1.35">'
            f'{_html.escape((g.get("phrase") or "")[:280])}</div>'
            f'<div>{chips}</div>'
            f'<div style="font-size:10px;color:rgba(10,50,53,0.5);margin-top:8px">'
            f'Left: clean &middot; Right: yellow=GT, ring=model pred (green clean_pos · '
            f'red clean_neg · gray ambiguous; no_pred hidden)</div>'
            f'</div></div>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per_bench', type=int, default=15, help='cards per bench')
    ap.add_argument('--seed', type=int, default=5)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    print('indexing pointing recs...', flush=True)
    pt_idx = {}
    for ds in PT_LOADERS:
        for r in PT_LOADERS[ds](f'{PT_BR}/{ds}'):
            pt_idx[f"{r['ds']}/{r['key']}"] = r
    print(f'  {len(pt_idx)}', flush=True)

    print('indexing reasonseg recs...', flush=True)
    rs_idx = {r['key']: r for r in load_reasonseg(RS_ROOT)}
    print(f'  {len(rs_idx)}', flush=True)

    print('re-sampling filter recs (deterministic, same seed as eval)...', flush=True)
    fl_idx = {}
    for ds in FLT_SUB:
        for r in flt_sample(ds, 2000, 20):
            fl_idx[f"{ds}/{r['tar']}/{r['key_in_tar']}"] = (ds, r)
    print(f'  {len(fl_idx)}', flush=True)

    pool = [json.loads(l) for l in open(POOL_PATH)]
    clean = [g for g in pool if g.get('clean_pos') and g.get('clean_neg')]
    print(f'pool {len(pool)} | clean pairs {len(clean)}', flush=True)

    # per-bench sample, grouped sections in the panel
    by_bench = {'pointing': [], 'reasonseg': [], 'filter': []}
    for g in clean:
        by_bench.setdefault(g['bench'], []).append(g)
    BENCH_TITLE = {'pointing': 'Pointing benchmarks (pointarena / refspatial / robospatial / where2place)',
                   'reasonseg': 'ReasonSeg',
                   'filter': 'Filter datasets (vg / openimages / pixmo / grit / cc3m / cc12m / seeclick / rf100, sampled)'}
    sections = []
    total_cards = 0
    for bench in ['pointing', 'reasonseg', 'filter']:
        pool_b = by_bench.get(bench, [])
        picks = rng.sample(pool_b, min(args.per_bench, len(pool_b)))
        bcards = []
        for g in picks:
            c = render_card(g, (pt_idx, rs_idx, fl_idx))
            if c: bcards.append(c)
        total_cards += len(bcards)
        sections.append(
            f'<h3 style="margin:24px 0 4px;color:#105257;border-bottom:1px solid rgba(10,50,53,0.15);'
            f'padding-bottom:4px">{BENCH_TITLE[bench]} &mdash; <span style="font-weight:400;'
            f'color:rgba(10,50,53,0.6);font-size:14px">{len(pool_b)} clean pairs, showing {len(bcards)}'
            f'</span></h3>\n' + '\n'.join(bcards))
        print(f'  {bench}: clean {len(pool_b)} -> rendered {len(bcards)}', flush=True)
    cards_html = '\n'.join(sections)

    n_pred = sum(len(g['preds']) for g in pool)
    n_pos = sum(sum(p['label'] == 'pos' for p in g['preds']) for g in pool)
    n_cp = sum(sum(p.get('quality') == 'clean_pos' for p in g['preds']) for g in pool)
    n_cn = sum(sum(p.get('quality') == 'clean_neg' for p in g['preds']) for g in pool)
    n_amb = sum(sum(p.get('quality') == 'ambiguous' for p in g['preds']) for g in pool)
    n_np = sum(sum(p.get('quality') == 'no_pred' for p in g['preds']) for g in pool)

    intro = (
        '<div class="dataset-intro">'
        '<div class="intro-title"><b>Model-spectrum harvest pool &mdash; REAL grounding errors '
        '(consensus-verified)</b></div>'
        f'<div class="intro-desc">Seven pointing models (weak&rarr;strong, 3 architectures: '
        f'MolmoE-1B / Molmo-7B / Molmo2-4B / Qwen2.5-VL-3B+7B+32B / InternVL3-8B) run on '
        f'pointing&nbsp;1404 + ReasonSeg&nbsp;196 + filter&nbsp;~16000 = {len(pool):,} (image, phrase) groups. '
        f'Each prediction is scored vs GT, then quality-filtered by <b>cross-model consensus</b>: '
        f'a clean_pos is GT-hit AND near consensus; a clean_neg is GT-miss AND far from consensus '
        f'(real model error, not a GT-too-strict false-rejection).</div>'
        f'<div class="intro-meta">{n_pred:,} predictions ({n_pos:,} GT-hit / {n_pred-n_pos:,} GT-miss) '
        f'&middot; quality: clean_pos {n_cp:,} &middot; clean_neg {n_cn:,} &middot; ambiguous {n_amb:,} '
        f'&middot; no_pred {n_np:,} &middot; <b>{len(clean):,} CLEAN preference pairs &mdash; '
        f'directly usable RM training data</b></div></div>')

    panel = (f'<div id="p_3_spectrum_pool" class="panel" data-cat="cat3">\n{intro}\n'
             f'<div style="font-size:11px;color:rgba(10,50,53,0.55);margin:8px 0">'
             f'Each card: <b>left</b> = clean image &middot; <b>right</b> = same image with '
             f'yellow GT and each model\'s prediction (ring + short tag: M2=Molmo2-4B, M7=Molmo-7B, '
             f'ME=MolmoE-1B, Q3=Qwen2.5VL-3B, Q7=Qwen2.5VL-7B, Q32=Qwen2.5VL-32B, IV=InternVL3-8B; green=clean_pos, '
             f'red=clean_neg, gray=ambiguous; no_pred hidden). Showing {total_cards} sampled clean pairs.</div>\n'
             + cards_html + '\n</div>\n')
    tab = '<button class="ds-tab" data-panel="p_3_spectrum_pool">Spectrum Pool</button>'

    os.makedirs(SNIP, exist_ok=True)
    with open(os.path.join(SNIP, 'cat3_judge_tab.html'), 'w') as f: f.write(tab)
    with open(os.path.join(SNIP, 'cat3_judge_panel.html'), 'w') as f: f.write(panel)
    print(f'wrote cat3_judge_tab.html + cat3_judge_panel.html ({len(panel):,}b, {total_cards} cards)')


if __name__ == '__main__':
    main()