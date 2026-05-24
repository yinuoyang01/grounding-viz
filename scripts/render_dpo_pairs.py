"""Render N random DPO preference pairs from rm_dpo_pairs.jsonl as a cat3 tab
("DPO Pairs") for human verification — does the chosen point truly belong to
the phrase? does the rejected point truly NOT?

Per card:
  left  = clean image
  right = same image with yellow GT + green ring (chosen) + red ring (rejected)
          and short model tags next to each dot
  meta  = bench/category, phrase, chosen model, rejected model, spatial gap
"""
import argparse
import base64
import io
import json
import os
import random
import sys
import math
import html as _html

sys.path.insert(0, '/weka/oe-training-default/zixianm/yinuoy/grounding_rm/data')
from eval_pointing import LOADERS as PT_LOADERS, BR as PT_BR  # noqa: E402
from eval_reasonseg import load_reasonseg  # noqa: E402
from eval_filter_sample import sample_ds as flt_sample, gt_pixel as flt_gt_px, SUB as FLT_SUB  # noqa: E402
from build_cat3_dataset_panels import resolve_tar, extract_for_ds  # noqa: E402
from judge_inline import DS_TO_TAR_DS  # noqa: E402

DPO_PATH = '/weka/oe-training-default/oe-encoder/rm_dpo_pairs.jsonl'
RS_ROOT = '/weka/oe-training-default/royg/grounding_data/reasonseg/val'
SNIP = '/weka/oe-training-default/zixianm/yinuoy/grounding-viz/.snippets'
SHORT = {'molmo2-4b': 'Molmo2-4B', 'molmo7b-d': 'Molmo-7B', 'molmoE-1b': 'MolmoE-1B',
         'qwen2.5vl-3b': 'Qwen2.5VL-3B', 'qwen2.5vl-7b': 'Qwen2.5VL-7B',
         'qwen2.5vl-32b': 'Qwen2.5VL-32B', 'internvl3-8b': 'InternVL3-8B'}


def _img_bytes(r):
    return open(r['img_path'], 'rb').read() if 'img_path' in r else r['img_bytes']


def render_card(pair, indexes):
    import numpy as np
    from PIL import Image, ImageDraw
    bench = pair['bench']
    pt_idx, rs_idx, fl_idx = indexes
    try:
        if bench == 'pointing':
            r = pt_idx.get(pair['item'])
            if not r: return None
            im = Image.open(io.BytesIO(_img_bytes(r))).convert('RGB'); W, H = im.size
            mb = open(r['mask_path'], 'rb').read() if 'mask_path' in r else r['mask_bytes']
            mk = Image.open(io.BytesIO(mb)).convert('L')
            if mk.size != (W, H): mk = mk.resize((W, H))
            gt = ('mask', np.array(mk))
        elif bench == 'reasonseg':
            r = rs_idx.get(pair['item'])
            if not r: return None
            im = Image.open(r['img_path']).convert('RGB'); W, H = im.size
            tmp = Image.new('L', (W, H), 0); td = ImageDraw.Draw(tmp)
            for poly in r['polys']:
                td.polygon([(int(p[0]), int(p[1])) for p in poly], fill=255)
            gt = ('mask', np.array(tmp))
        elif bench == 'filter':
            x = fl_idx.get(pair['item'])
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

    sc = 540 / max(W, H)
    nW, nH = max(1, int(W * sc)), max(1, int(H * sc))
    clean = im.resize((nW, nH))

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
    rad = 12
    # chosen = green, rejected = red
    spans = []
    for role, p, col in [('chosen', pair['chosen'], (0, 200, 80)),
                         ('rejected', pair['rejected'], (235, 40, 60))]:
        tag = SHORT.get(p['model'], p['model'])
        for (x, y) in (p.get('pred') or []):
            cx, cy = x * sc, y * sc
            d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad],
                      outline=(255, 255, 255), width=4)
            d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=col, width=4)
            tx, ty = cx + rad + 3, cy - rad - 2
            bb = d.textbbox((tx, ty), tag)
            d.rectangle([bb[0] - 2, bb[1] - 1, bb[2] + 2, bb[3] + 1], fill=(255, 255, 255))
            d.text((tx, ty), tag, fill=col)
            spans.append((role, x, y))

    # spatial gap (chosen vs rejected, in image px)
    gap = None
    if pair['chosen'].get('pred') and pair['rejected'].get('pred'):
        c0 = pair['chosen']['pred'][0]
        r0 = pair['rejected']['pred'][0]
        gap = math.hypot(c0[0] - r0[0], c0[1] - r0[1])

    def _b64(im):
        b = io.BytesIO(); im.save(b, 'JPEG', quality=83)
        return base64.b64encode(b.getvalue()).decode()
    bc, bm = _b64(clean), _b64(marked)

    img_diag = math.hypot(W, H)
    gap_str = (f'{gap:.0f} px  (img diag {img_diag:.0f}, {gap/img_diag*100:.1f}%)'
               if gap is not None else 'n/a')

    ST = ('display:grid;grid-template-columns:780px 1fr;gap:16px;padding:12px;margin:8px 0;'
          'background:var(--bg,#FAF2E9);border:1px solid rgba(10,50,53,0.15);border-radius:8px')
    chosen_m = _html.escape(SHORT.get(pair['chosen']['model'], pair['chosen']['model']))
    rejected_m = _html.escape(SHORT.get(pair['rejected']['model'], pair['rejected']['model']))
    return (f'<div style="{ST}">'
            f'<div style="display:flex;gap:8px">'
            f'<img src="data:image/jpeg;base64,{bc}" style="width:50%;border-radius:4px"/>'
            f'<img src="data:image/jpeg;base64,{bm}" style="width:50%;border-radius:4px"/>'
            f'</div>'
            f'<div>'
            f'<div style="color:rgba(10,50,53,0.55);font-family:ui-monospace,Menlo,monospace;'
            f'font-size:11px;margin-bottom:6px">{_html.escape(bench)} / '
            f'{_html.escape(str(pair.get("category","-")))}</div>'
            f'<div style="color:#0A3235;font-size:14px;font-weight:700;margin-bottom:10px;line-height:1.35">'
            f'{_html.escape((pair.get("phrase") or "")[:280])}</div>'
            f'<div style="font-size:12px;line-height:1.6">'
            f'<span style="background:#0FCB8C;color:#FAF2E9;padding:2px 7px;border-radius:4px;'
            f'font-weight:700">✓ chosen</span> &nbsp; <b>{chosen_m}</b><br/>'
            f'<span style="background:#F0529C;color:#FAF2E9;padding:2px 7px;border-radius:4px;'
            f'font-weight:700">✗ rejected</span> &nbsp; <b>{rejected_m}</b><br/>'
            f'<span style="color:rgba(10,50,53,0.6)">spatial gap: {gap_str}</span>'
            f'</div>'
            f'<div style="font-size:10px;color:rgba(10,50,53,0.5);margin-top:8px">'
            f'Left: clean &middot; Right: yellow=GT, green ring=chosen (clean_pos), '
            f'red ring=rejected (clean_neg)</div>'
            f'</div></div>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=100, help='pairs to render')
    ap.add_argument('--seed', type=int, default=7)
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

    pairs = [json.loads(l) for l in open(DPO_PATH)]
    print(f'pool {len(pairs)} pairs', flush=True)

    # stratified by bench so verification covers all 3 (not just filter)
    by_bench = {}
    for p in pairs:
        by_bench.setdefault(p['bench'], []).append(p)
    # split N proportionally but with a floor for under-represented benches
    floors = {'pointing': max(20, args.n // 5),
              'reasonseg': max(15, args.n // 7),
              'filter': args.n}  # fill remainder with filter
    picks = []
    for bench in ['pointing', 'reasonseg']:
        pool = by_bench.get(bench, [])
        k = min(floors[bench], len(pool))
        picks.extend(rng.sample(pool, k))
    remaining = args.n - len(picks)
    if remaining > 0:
        pool = by_bench.get('filter', [])
        k = min(remaining, len(pool))
        picks.extend(rng.sample(pool, k))
    rng.shuffle(picks)
    print(f'rendering {len(picks)} pairs (pointing/reasonseg/filter stratified)', flush=True)

    cards = []
    fails = 0
    for pair in picks:
        c = render_card(pair, (pt_idx, rs_idx, fl_idx))
        if c:
            cards.append(c)
        else:
            fails += 1
    print(f'  rendered {len(cards)} / failed {fails}', flush=True)

    # bench counts in actually-rendered set
    counts = {}
    rendered_benches = []
    for i, pair in enumerate(picks):
        if i < len(cards):
            rendered_benches.append(pair['bench'])
    for b in rendered_benches:
        counts[b] = counts.get(b, 0) + 1
    counts_str = ' &middot; '.join(f'{b}: {n}' for b, n in counts.items())

    intro = (
        '<div class="dataset-intro">'
        '<div class="intro-title"><b>DPO preference pairs &mdash; human verification view</b></div>'
        f'<div class="intro-desc">{len(cards)} random pairs (stratified across pointing / reasonseg / filter) '
        f'from the {len(pairs):,}-pair RM training pool. Each card shows ONE preference pair: '
        f'<b style="color:#0A8A4E">green ring = chosen (clean_pos)</b>, '
        f'<b style="color:#B22D45">red ring = rejected (clean_neg)</b>. '
        f'Use this view to spot-check whether chosen really matches the phrase '
        f'and rejected really doesn\'t.</div>'
        f'<div class="intro-meta">rendered: {counts_str}</div></div>')

    panel = (f'<div id="p_3_dpo_pairs" class="panel" data-cat="cat3">\n{intro}\n'
             + '\n'.join(cards) + '\n</div>\n')
    tab = '<button class="ds-tab" data-panel="p_3_dpo_pairs">DPO Pairs</button>'

    os.makedirs(SNIP, exist_ok=True)
    with open(os.path.join(SNIP, 'cat3_dpo_pairs_tab.html'), 'w') as f: f.write(tab)
    with open(os.path.join(SNIP, 'cat3_dpo_pairs_panel.html'), 'w') as f: f.write(panel)
    print(f'wrote cat3_dpo_pairs_tab.html + cat3_dpo_pairs_panel.html '
          f'({len(panel):,}b, {len(cards)} cards)')


if __name__ == '__main__':
    main()
