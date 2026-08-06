"""Render GroundCUA samples to a standalone browse page.

GroundCUA is the candidate fix for ScreenSpot-Pro, where every reward we tried lands within
noise of base (0.1398 base vs 0.1404 GT-RLVR, 51/50 paired, p=1.00). The training pool has one
GUI source, SeeClick, and it is web/mobile at <=1920x1080; ScreenSpot-Pro is desktop
professional software at 3456x1440 median with targets ~20x smaller. GroundCUA sits between
them: 86 desktop apps, 1920x1009 median, targets 0.123% of image area.

This page exists to eyeball whether the phrases and boxes are usable BEFORE building any
training data from it, so the panel deliberately shows the raw record: the click point the
dataset ships, its bbox, and the question text verbatim.

Coordinate convention (verified, not guessed): metadata.bbox and answer.x/answer.y are BOTH
percentages of image size, 0-100, xyxy order. 2999/3000 sampled clicks fall inside their own
bbox, which is what pins the convention down.
"""
import argparse
import base64
import html
import io
import json
import os
import random

from PIL import Image, ImageDraw

ROOT = '/weka/oe-training-default/webolmo/datasets/GroundCUA'
DATA = f'{ROOT}/formatted_data.json'

TEAL = (16, 82, 87)
PINK = (240, 82, 156)
GREEN = (15, 203, 140)


def render(rec, max_w=900, zoom_pad=0.06):
    """-> (full-view b64, zoom-crop b64, W, H, target area %). Two views because a
    0.1%-area target is invisible at page width; the crop is what a human needs to judge
    whether the phrase actually identifies that widget."""
    im = Image.open(rec['image']).convert('RGB')
    W, H = im.size
    b = rec['metadata']['bbox']
    x1, y1, x2, y2 = b[0] / 100 * W, b[1] / 100 * H, b[2] / 100 * W, b[3] / 100 * H
    a = json.loads(rec['answer'])
    px, py = a['x'] / 100 * W, a['y'] / 100 * H

    full = im.copy()
    d = ImageDraw.Draw(full)
    lw = max(2, int(0.002 * max(W, H)))
    d.rectangle([x1, y1, x2, y2], outline=GREEN, width=lw)
    r = max(4, int(0.004 * max(W, H)))
    d.ellipse([px - r, py - r, px + r, py + r], fill=PINK, outline=(255, 255, 255), width=2)

    pad = zoom_pad * max(W, H)
    crop = full.crop((max(0, x1 - pad), max(0, y1 - pad),
                      min(W, x2 + pad), min(H, y2 + pad)))

    def b64(img, width):
        img = img.copy()
        if img.width > width:
            img = img.resize((width, max(1, int(img.height * width / img.width))), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, 'JPEG', quality=82)
        return base64.b64encode(buf.getvalue()).decode()

    area = (x2 - x1) * (y2 - y1) / (W * H) * 100
    return b64(full, max_w), b64(crop, 520), W, H, area


ST_CARD = ('display:grid;grid-template-columns:1fr 540px;gap:18px;padding:14px;margin:10px 0;'
           'background:#F1E4D1;border:1px solid rgba(10,50,53,0.15);border-radius:8px')
ST_META = ('color:rgba(10,50,53,0.55);font-family:ui-monospace,Menlo,monospace;'
           'font-size:11px;margin-bottom:6px')
ST_TITLE = ('font-weight:700;color:#105257;font-size:11px;margin-top:10px;'
            'text-transform:uppercase;letter-spacing:.04em')
ST_TEXT = 'color:#0A3235;font-size:13px;margin-top:2px'


def card(rec, i):
    full, crop, W, H, area = render(rec)
    app = rec['image'].split('/images/')[1].split('/')[0]
    q = html.escape(rec['question'][:400])
    a = json.loads(rec['answer'])
    return (
        f'<div style="{ST_CARD}">\n'
        f'  <div><img src="data:image/jpeg;base64,{full}" style="width:100%;border-radius:4px"/>'
        f'<div style="{ST_META};margin-top:6px">full screenshot {W}&times;{H}</div></div>\n'
        f'  <div>\n'
        f'    <div style="{ST_META}">#{i} &nbsp; {html.escape(app)}</div>\n'
        f'    <img src="data:image/jpeg;base64,{crop}" style="width:100%;border-radius:4px;'
        f'border:1px solid rgba(10,50,53,0.2)"/>\n'
        f'    <div style="{ST_META};margin-top:6px">zoomed to the target</div>\n'
        f'    <div style="{ST_TITLE}">instruction</div>\n'
        f'    <div style="{ST_TEXT}"><b>{q}</b></div>\n'
        f'    <div style="{ST_TITLE}">target</div>\n'
        f'    <div style="{ST_TEXT}">area <b>{area:.3f}%</b> of image &nbsp;·&nbsp; '
        f'click ({a["x"]:.1f}, {a["y"]:.1f})%</div>\n'
        f'    <div style="{ST_TITLE}">overlay</div>\n'
        f'    <div style="{ST_TEXT};font-size:11px">'
        f'<span style="color:#0FCB8C;font-weight:600">▭</span> GT bbox &nbsp;'
        f'<span style="color:#F0529C;font-weight:600">●</span> GT click point</div>\n'
        f'  </div>\n</div>\n')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=5)
    ap.add_argument('--seed', type=int, default=1)
    ap.add_argument('--out', default=os.path.join(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))), 'groundcua.html'))
    ap.add_argument('--one_per_app', action='store_true',
                    help='spread the sample across distinct apps instead of drawing uniformly '
                         '(the pool is dominated by RStudio/Frappe Books, so a uniform draw '
                         'shows the same app repeatedly)')
    args = ap.parse_args()

    d = json.load(open(DATA))['data']
    print(f'GroundCUA rows: {len(d):,}', flush=True)
    rng = random.Random(args.seed)
    rng.shuffle(d)
    picks, seen = [], set()
    for r in d:
        app = r['image'].split('/images/')[1].split('/')[0]
        if args.one_per_app and app in seen:
            continue
        if not os.path.isfile(r['image']):
            continue
        seen.add(app)
        picks.append(r)
        if len(picks) >= args.n:
            break

    cards, err = [], 0
    for i, r in enumerate(picks, 1):
        try:
            cards.append(card(r, i))
        except Exception as e:
            err += 1
            print(f'  render fail #{i}: {type(e).__name__}: {e}', flush=True)
    print(f'rendered {len(cards)}, err {err}', flush=True)

    page = f"""<!doctype html><meta charset="utf-8"><title>GroundCUA samples</title>
<body style="margin:0;background:#FAF2E9;color:#0A3235;
font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif">
<div style="max-width:1600px;margin:0 auto;padding:28px">
<h1 style="color:#0A3235;margin:0 0 4px">GroundCUA — sample records</h1>
<div style="color:rgba(10,50,53,0.65);font-size:14px;max-width:900px;line-height:1.5">
3,131,480 records over 51,175 screenshots from 86 desktop applications (Blender, FreeCAD, GIMP,
Eclipse, RStudio, Scribus, …), median 1920&times;1009, target median 0.123% of image area.
Candidate training source for the ScreenSpot-Pro gap: the current pool's only GUI data is
SeeClick (web/mobile, &le;1920&times;1080), while ScreenSpot-Pro is desktop professional software
at 3456&times;1440 median with targets ~20&times; smaller.
Coordinates are percentages (0&ndash;100) for both bbox and click; 2999/3000 sampled clicks fall
inside their own bbox.
</div>
<div style="margin-top:18px">
{''.join(cards)}
</div></div></body>"""
    with open(args.out, 'w') as f:
        f.write(page)
    print(f'wrote {args.out} ({os.path.getsize(args.out)/1e6:.1f} MB)')


if __name__ == '__main__':
    main()
