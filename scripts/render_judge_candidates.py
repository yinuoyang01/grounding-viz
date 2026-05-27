"""Render judge-candidate samples for human eyeball QA.

Per card:
  left  = clean image
  right = image with YELLOW translucent GT shape(s) + GREEN ring at consensus center
          + RED dots at each strong-model pred (with model tag)
  meta  = ds / phrase / n_gt / consensus_n_strong / dist_over_tol

Lets us spot-check whether the GT-incomplete candidate criteria actually catch real
'GT-missed-an-instance' cases before paying for VLM judge at scale.
"""
import argparse
import base64
import html as _html
import io
import json
import math
import os
import random
import sys

sys.path.insert(0, '/weka/oe-training-default/zixianm/yinuoy/grounding_rm/data')
sys.path.insert(0, '/weka/oe-training-default/zixianm/yinuoy/grounding-viz/scripts')
from build_cat3_dataset_panels import resolve_tar, extract_for_ds
from judge_inline import DS_TO_TAR_DS

SNIP = '/weka/oe-training-default/zixianm/yinuoy/grounding-viz/.snippets'
SHORT = {'qwen2.5vl-32b': 'Qwen2.5VL-32B', 'qwen2.5vl-7b': 'Qwen2.5VL-7B',
         'molmo7b-d': 'Molmo-7B', 'internvl3-8b': 'InternVL3-8B'}


def render_card(c):
    from PIL import Image, ImageDraw
    ds = c['ds']
    item = c['item']
    tar_ds = DS_TO_TAR_DS.get(ds, ds)
    tar_name = item.split('/')[1]
    key_in_tar = item.split('/', 2)[2]
    try:
        tp = resolve_tar(tar_ds, tar_name)
        if not tp:
            return None
        ib = extract_for_ds(tar_ds, tp, key_in_tar)
        if not ib:
            return None
        im = Image.open(io.BytesIO(ib)).convert('RGB')
    except Exception:
        return None
    W, H = im.size

    sc = 540 / max(W, H)
    nW, nH = max(1, int(W * sc)), max(1, int(H * sc))
    clean = im.resize((nW, nH))

    marked_full = im.copy()
    d0 = ImageDraw.Draw(marked_full)
    lw = max(3, W // 280)
    for typ, geom in c['gt_shapes']:
        if typ == 'bbox':
            d0.rectangle([min(geom[0], geom[2]), min(geom[1], geom[3]),
                          max(geom[0], geom[2]), max(geom[1], geom[3])],
                         outline=(255, 210, 0), width=lw)
            xs = [geom[0], geom[2]]; ys = [geom[1], geom[3]]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            overlay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
            ImageDraw.Draw(overlay).rectangle([x0, y0, x1, y1], fill=(255, 210, 0, 60))
            marked_full = Image.alpha_composite(marked_full.convert('RGBA'), overlay).convert('RGB')
            d0 = ImageDraw.Draw(marked_full)
        else:
            rr = max(8, W // 120)
            d0.ellipse([geom[0] - rr, geom[1] - rr, geom[0] + rr, geom[1] + rr],
                       outline=(255, 210, 0), width=lw)
    marked = marked_full.resize((nW, nH))
    d = ImageDraw.Draw(marked)

    cx, cy = c['consensus_center']
    cx *= sc; cy *= sc
    rad = 18
    d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=(255, 255, 255), width=5)
    d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=(0, 200, 80), width=4)

    rad2 = 7
    for sp in c['strong_preds']:
        px, py = sp['cx'] * sc, sp['cy'] * sc
        d.ellipse([px - rad2, py - rad2, px + rad2, py + rad2],
                  outline=(255, 255, 255), width=3)
        d.ellipse([px - rad2, py - rad2, px + rad2, py + rad2], fill=(235, 40, 60))
        tag = SHORT.get(sp['model'], sp['model'])
        tx, ty = px + rad2 + 3, py - rad2 - 2
        bb = d.textbbox((tx, ty), tag)
        d.rectangle([bb[0] - 2, bb[1] - 1, bb[2] + 2, bb[3] + 1], fill=(255, 255, 255))
        d.text((tx, ty), tag, fill=(235, 40, 60))

    def _b64(im):
        b = io.BytesIO(); im.save(b, 'JPEG', quality=83)
        return base64.b64encode(b.getvalue()).decode()
    bc, bm = _b64(clean), _b64(marked)

    img_diag = math.hypot(W, H)
    dot = c['dist_over_tol']
    dist_px = c['consensus_dist_to_gt']
    dist_str = f'{dist_px:.0f} px ({dist_px/img_diag*100:.1f}% img diag) · {dot:.1f}×tol'

    ST = ('display:grid;grid-template-columns:780px 1fr;gap:16px;padding:12px;margin:8px 0;'
          'background:var(--bg,#FAF2E9);border:1px solid rgba(10,50,53,0.15);border-radius:8px')
    return (f'<div style="{ST}">'
            f'<div style="display:flex;gap:8px">'
            f'<img src="data:image/jpeg;base64,{bc}" style="width:50%;border-radius:4px"/>'
            f'<img src="data:image/jpeg;base64,{bm}" style="width:50%;border-radius:4px"/>'
            f'</div>'
            f'<div>'
            f'<div style="color:rgba(10,50,53,0.55);font-family:ui-monospace,Menlo,monospace;'
            f'font-size:11px;margin-bottom:6px">{_html.escape(ds)}</div>'
            f'<div style="color:#0A3235;font-size:14px;font-weight:700;margin-bottom:10px;line-height:1.35">'
            f'{_html.escape((c.get("phrase") or "")[:280])}</div>'
            f'<div style="font-size:12px;line-height:1.6">'
            f'<span style="background:#FFCC00;color:#0A3235;padding:2px 7px;border-radius:4px;'
            f'font-weight:700">GT</span> &nbsp; n_gt={c["n_gt"]}<br/>'
            f'<span style="background:#0FCB8C;color:#FAF2E9;padding:2px 7px;border-radius:4px;'
            f'font-weight:700">○ consensus</span> &nbsp; {c["consensus_n_strong"]}/4 strong models<br/>'
            f'<span style="color:rgba(10,50,53,0.7)">distance to GT: {dist_str}</span>'
            f'</div>'
            f'<div style="font-size:10px;color:rgba(10,50,53,0.5);margin-top:8px">'
            f'Left: clean &middot; Right: yellow=GT region, green ring=strong-model consensus, '
            f'red dots=individual strong preds. If consensus location is also a valid match for '
            f'phrase, GT is incomplete &rarr; consensus_only candidate for RM positives.</div>'
            f'</div></div>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidates', default='/weka/oe-training-default/oe-encoder/judge_candidates_v36.jsonl')
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--suffix', default='_v36')
    ap.add_argument('--title', default='Judge candidates v36 (GT-incomplete suspects)')
    args = ap.parse_args()

    cands = [json.loads(l) for l in open(args.candidates)]
    print(f'pool {len(cands)} candidates', flush=True)

    rng = random.Random(args.seed)
    picks = rng.sample(cands, min(args.n, len(cands)))

    cards = []
    fails = 0
    for c in picks:
        card = render_card(c)
        if card:
            cards.append(card)
        else:
            fails += 1
    print(f'rendered {len(cards)} / failed {fails}', flush=True)

    by_ds = {}
    for c, ok in zip(picks, [True] * len(cards) + [False] * fails):
        if ok:
            by_ds[c['ds']] = by_ds.get(c['ds'], 0) + 1
    counts_str = ' &middot; '.join(f'{ds}: {n}' for ds, n in by_ds.items())

    panel_id = f'p_3_judge_cand{args.suffix}'
    intro = (
        '<div class="dataset-intro">'
        f'<div class="intro-title"><b>{_html.escape(args.title)} &mdash; human eyeball QA</b></div>'
        f'<div class="intro-desc">{len(cards)} random candidates (out of {len(cands):,} total) '
        f'where: n_gt=1, no clean_pos, &geq;3 of 4 strong models cluster, consensus &gt;2&times;tol '
        f'from GT center. Each card: <b style="color:#0A8A4E">green ring</b> = strong-model consensus, '
        f'<b style="color:#B22D45">red dots</b> = strong preds, <b style="color:#B58800">yellow</b> = GT. '
        f'Question: does the green ring location also match the phrase? If yes &rarr; GT is incomplete, '
        f'candidate for RM positive backfill.</div>'
        f'<div class="intro-meta">rendered: {counts_str}</div></div>')

    panel = (f'<div id="{panel_id}" class="panel" data-cat="cat3">\n{intro}\n'
             + '\n'.join(cards) + '\n</div>\n')
    tab = f'<button class="ds-tab" data-panel="{panel_id}">{_html.escape(args.title)}</button>'

    os.makedirs(SNIP, exist_ok=True)
    tab_name = f'cat3_judge_cand{args.suffix}_tab.html'
    panel_name = f'cat3_judge_cand{args.suffix}_panel.html'
    with open(os.path.join(SNIP, tab_name), 'w') as f: f.write(tab)
    with open(os.path.join(SNIP, panel_name), 'w') as f: f.write(panel)
    print(f'wrote {tab_name} + {panel_name} ({len(panel):,}b, {len(cards)} cards)')


if __name__ == '__main__':
    main()
