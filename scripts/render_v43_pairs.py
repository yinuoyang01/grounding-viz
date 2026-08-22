"""Render v43 RM pair samples (SAM2-verified placement tiers + extra1_near) to
v43_pairs.html for human verification of the offset construction.

Tiers are relative to each image's min pairwise GT spacing (d_min):
near_hard 0.45-0.85x, near_mid 0.85-1.6x, near_far 1.6-3.0x; extra1_near adds one
wrong mark just off a GT point (fewer-accurate-beats-more direction). All pairs
shown here survived SAM2 point-prompted mask verification.

    python3 scripts/render_v43_pairs.py --per_variant 6
"""
import argparse
import base64
import glob
import html as _html
import io
import json
import random

from PIL import Image

G = '/weka/oe-training-default/zixianm/yinuoy/grounding_rm/data'
OUT = '/weka/oe-training-default/zixianm/yinuoy/grounding-viz/v43_pairs.html'

DESC = {
    'v43_near_hard': 'near_hard — m marks offset 0.45-0.85 x GT spacing (boundary graze)',
    'v43_near_mid': 'near_mid — 0.85-1.6 x (clear miss, same region)',
    'v43_near_far': 'near_far — 1.6-3.0 x (wrong area)',
    'v43_extra1_near': 'extra1_near — k correct + 1 wrong mark just off a GT point (fewer-wins)',
}


def b64(im, max_dim, q=72):
    im = im.copy()
    im.thumbnail((max_dim, max_dim))
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=q)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per_variant', type=int, default=6)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    byv = {}
    for f in sorted(glob.glob(f'{G}/som_synth_v43/train.verified.jsonl*')):
        for line in open(f):
            r = json.loads(line)
            byv.setdefault(r['meta']['pool'], []).append(r)

    cards = []
    for pool in ('v43_near_hard', 'v43_near_mid', 'v43_near_far', 'v43_extra1_near'):
        rows = byv.get(pool, [])
        cards.append(f'<h2>{_html.escape(DESC.get(pool, pool))} <span style="font-weight:normal;'
                     f'font-size:14px;color:#666">({len(rows)} verified pairs)</span></h2>')
        for r in rng.sample(rows, min(args.per_variant, len(rows))):
            ci = 0 if r['label'] == 0 else 1
            m = r['meta']
            try:
                imc = Image.open(r['images'][ci]).convert('RGB')
                imr = Image.open(r['images'][1 - ci]).convert('RGB')
            except Exception:
                continue
            title = (f"“{_html.escape(str(m.get('phrase'))[:110])}” "
                     f"| k={m.get('pts_chosen')} → k_rej={m.get('pts_rejected')} "
                     f"| margin={m.get('margin')}")
            cards.append(
                '<div style="border:1px solid #ccc;border-radius:8px;padding:10px;margin:14px 0">'
                f'<div style="font-size:13px;margin-bottom:6px">{title}</div>'
                '<div style="display:flex;gap:8px">'
                f'<div style="flex:1"><img style="width:100%" src="data:image/jpeg;base64,{b64(imc, 820)}">'
                '<div style="font-size:12px;color:#080"><b>CHOSEN</b> — marks on GT points</div></div>'
                f'<div style="flex:1"><img style="width:100%" src="data:image/jpeg;base64,{b64(imr, 820)}">'
                '<div style="font-size:12px;color:#a00"><b>REJECTED</b> — offset/extra marks</div></div></div>'
                '</div>')

    page = ('<!DOCTYPE html><html><head><meta charset="utf-8">'
            '<title>v43 RM pairs — samples</title></head>'
            '<body style="font-family:sans-serif;max-width:1200px;margin:0 auto;padding:20px">'
            '<h1>v43 RM pairs — placement tiers (SAM2-verified)</h1>'
            '<p>Same som_marks_v2 renders and v42-numbered text contract as before; only the '
            'rejected-mark placement is new. Offsets are relative to each image\'s min GT '
            'spacing; every shown pair passed SAM2 verification that the wrong marks land '
            'outside the object masks.</p>'
            + '\n'.join(cards) + '</body></html>')
    with open(OUT, 'w') as f:
        f.write(page)
    print(f'{OUT}: {len(cards)} blocks, {len(page)/1e6:.1f} MB')


if __name__ == '__main__':
    main()
