"""Render v41 RM pair samples (pixmo exact-count triplets + GUISyn negatives) to
v41_pairs.html for human verification of the som_marks_v2 rendering and pair logic.

Per card: chosen render | rejected render, and for GUI pairs an extra native-res crop
around the marks (4K screenshots downscale the badges out of visibility otherwise).

    python3 scripts/render_v41_pairs.py --per_variant 5
"""
import argparse
import base64
import html as _html
import io
import json
import random

from PIL import Image

G = '/weka/oe-training-default/zixianm/yinuoy/grounding_rm/data'
SOURCES = [(f'{G}/som_synth_v41', 'pixmo'), (f'{G}/som_synth_v41_gui', 'gui')]
OUT = '/weka/oe-training-default/zixianm/yinuoy/grounding-viz/v41_pairs.html'


def b64(im, max_dim, q=72):
    im = im.copy()
    im.thumbnail((max_dim, max_dim))
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=q)
    return base64.b64encode(buf.getvalue()).decode()


def crop_around(im, pts, size=760):
    if not pts:
        return None
    cx = sum(p[0] for p in pts) / len(pts)
    cy = sum(p[1] for p in pts) / len(pts)
    W, H = im.size
    x0 = max(0, min(int(cx) - size // 2, W - size))
    y0 = max(0, min(int(cy) - size // 2, H - size))
    return im.crop((x0, y0, min(x0 + size, W), min(y0 + size, H)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per_variant', type=int, default=5)
    ap.add_argument('--seed', type=int, default=7)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    cards = []
    for src, tag in SOURCES:
        rows = [json.loads(l) for l in open(f'{src}/train.jsonl')]
        byv = {}
        for r in rows:
            byv.setdefault(r['meta']['pool'], []).append(r)
        for pool in sorted(byv):
            for r in rng.sample(byv[pool], min(args.per_variant, len(byv[pool]))):
                ci = 0 if r['label'] == 0 else 1
                m = r['meta']
                try:
                    imc = Image.open(r['images'][ci]).convert('RGB')
                    imr = Image.open(r['images'][1 - ci]).convert('RGB')
                except Exception:
                    continue
                extra = ''
                if tag == 'gui':
                    pts = (m.get('pts_chosen') or []) + (m.get('pts_rejected') or [])
                    cc = crop_around(imc, m.get('pts_chosen') or pts)
                    cr = crop_around(imr, m.get('pts_rejected') or pts)
                    if cc is not None and cr is not None:
                        extra = (
                            '<div style="display:flex;gap:8px;margin-top:4px">'
                            f'<div style="flex:1"><img style="width:100%" src="data:image/jpeg;base64,{b64(cc, 760, 80)}">'
                            '<div style="font-size:11px;color:#080">chosen — native-res crop</div></div>'
                            f'<div style="flex:1"><img style="width:100%" src="data:image/jpeg;base64,{b64(cr, 760, 80)}">'
                            '<div style="font-size:11px;color:#a00">rejected — native-res crop</div></div></div>')
                margin = m.get('f1_margin')
                title = (f"<b>{_html.escape(pool)}</b> | “{_html.escape(str(m.get('phrase'))[:110])}” "
                         f"| k={m.get('k')} → k_rej={m.get('k_rej')}"
                         + (f" | margin={margin}" if margin is not None else ''))
                cards.append(
                    '<div style="border:1px solid #ccc;border-radius:8px;padding:10px;margin:14px 0">'
                    f'<div style="font-size:13px;margin-bottom:6px">{title}</div>'
                    '<div style="display:flex;gap:8px">'
                    f'<div style="flex:1"><img style="width:100%" src="data:image/jpeg;base64,{b64(imc, 820)}">'
                    '<div style="font-size:12px;color:#080"><b>CHOSEN</b></div></div>'
                    f'<div style="flex:1"><img style="width:100%" src="data:image/jpeg;base64,{b64(imr, 820)}">'
                    '<div style="font-size:12px;color:#a00"><b>REJECTED</b></div></div></div>'
                    f'{extra}</div>')

    page = ('<!DOCTYPE html><html><head><meta charset="utf-8">'
            '<title>v41 RM pairs — samples</title></head>'
            '<body style="font-family:sans-serif;max-width:1200px;margin:0 auto;padding:20px">'
            '<h1>v41 RM pairs — som_marks_v2 renders</h1>'
            '<p>pixmo exact-count triplets (chosen = exactly the k GT points) and GUISyn '
            'single-point negatives. GUI cards add native-res crops around the marks.</p>'
            + '\n'.join(cards) + '</body></html>')
    with open(OUT, 'w') as f:
        f.write(page)
    print(f'{OUT}: {len(cards)} cards, {len(page)/1e6:.1f} MB')


if __name__ == '__main__':
    main()
