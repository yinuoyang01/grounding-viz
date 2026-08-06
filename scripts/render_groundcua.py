"""Add a GroundCUA tab to the "Grounding data filtering" (cat3) section.

GroundCUA is the candidate fix for ScreenSpot-Pro, where every reward we tried lands within
noise of base (0.1398 base vs 0.1404 GT-RLVR, 51/50 paired, p=1.00). The training pool's only
GUI source is SeeClick (web/mobile, <=1920x1080); ScreenSpot-Pro is desktop professional
software at 3456x1440 median with targets ~20x smaller. GroundCUA sits between them: 86 desktop
apps, 1920x1009 median, targets 0.123% of image area.

Unlike the other cat3 tabs this one shows the RAW dataset record (its shipped click point and
bbox), not a Molmo prediction: the question here is whether the source data is usable at all,
before anything is trained on it.

Coordinate convention (verified, not guessed): metadata.bbox and answer.x/answer.y are BOTH
percentages of image size, 0-100, xyxy order. 2999/3000 sampled clicks fall inside their own
bbox, which is what pins the convention down.

Writes .snippets/cat3_groundcua_{tab,panel}.html and patches index.html in place, so the tab
survives whether or not generate.py is re-run.
"""
import argparse
import base64
import html
import io
import json
import os
import random
import re

from PIL import Image, ImageDraw

ROOT = '/weka/oe-training-default/webolmo/datasets/GroundCUA'
DATA = f'{ROOT}/formatted_data.json'
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

PINK = (240, 82, 156)
GREEN = (15, 203, 140)


def render(rec, full_w=720, crop_w=420, zoom_pad=0.06):
    """-> (full b64, zoom b64, W, H, area%). Two views: a 0.1%-area widget is invisible at
    page width, so the crop is what actually lets a human check the phrase against the box."""
    im = Image.open(rec['image']).convert('RGB')
    W, H = im.size
    b = rec['metadata']['bbox']
    x1, y1, x2, y2 = b[0] / 100 * W, b[1] / 100 * H, b[2] / 100 * W, b[3] / 100 * H
    a = json.loads(rec['answer'])
    px, py = a['x'] / 100 * W, a['y'] / 100 * H

    full = im.copy()
    d = ImageDraw.Draw(full)
    d.rectangle([x1, y1, x2, y2], outline=GREEN, width=max(2, int(0.002 * max(W, H))))
    r = max(4, int(0.004 * max(W, H)))
    d.ellipse([px - r, py - r, px + r, py + r], fill=PINK, outline=(255, 255, 255), width=2)

    pad = zoom_pad * max(W, H)
    crop = full.crop((max(0, x1 - pad), max(0, y1 - pad), min(W, x2 + pad), min(H, y2 + pad)))

    def b64(img, width):
        img = img.copy()
        if img.width > width:
            img = img.resize((width, max(1, int(img.height * width / img.width))), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, 'JPEG', quality=82)
        return base64.b64encode(buf.getvalue()).decode()

    return b64(full, full_w), b64(crop, crop_w), W, H, (x2 - x1) * (y2 - y1) / (W * H) * 100


ST_ROW = ('display:grid;grid-template-columns:720px 1fr;gap:16px;padding:12px;margin:8px 0;'
          'background:var(--bg,#FAF2E9);border:1px solid rgba(10,50,53,0.15);border-radius:8px')
ST_META = ('color:rgba(10,50,53,0.55);font-family:ui-monospace,Menlo,monospace;font-size:11px')
ST_TITLE = ('font-weight:700;color:#105257;font-size:11px;margin-top:10px;'
            'text-transform:uppercase;letter-spacing:.04em')
ST_TEXT = 'color:#0A3235;font-size:13px;margin-top:2px'


def card(rec, i):
    full, crop, W, H, area = render(rec)
    app = rec['image'].split('/images/')[1].split('/')[0]
    a = json.loads(rec['answer'])
    return (
        f'<div style="{ST_ROW}">'
        f'<div><img src="data:image/jpeg;base64,{full}" style="width:100%;border-radius:4px"/>'
        f'<div style="{ST_META};margin-top:6px">full screenshot {W}&times;{H}</div></div>'
        f'<div>'
        f'<div style="{ST_META}">#{i} &nbsp; {html.escape(app)}</div>'
        f'<img src="data:image/jpeg;base64,{crop}" style="width:100%;max-width:420px;'
        f'border-radius:4px;border:1px solid rgba(10,50,53,0.2);margin-top:4px"/>'
        f'<div style="{ST_META};margin-top:4px">zoomed to the target</div>'
        f'<div style="{ST_TITLE}">instruction</div>'
        f'<div style="{ST_TEXT}"><b>{html.escape(rec["question"][:400])}</b></div>'
        f'<div style="{ST_TITLE}">target</div>'
        f'<div style="{ST_TEXT}">area <b>{area:.3f}%</b> of image &nbsp;·&nbsp; '
        f'click ({a["x"]:.1f}, {a["y"]:.1f})%</div>'
        f'<div style="{ST_TITLE}">overlay</div>'
        f'<div style="{ST_TEXT};font-size:11px">'
        f'<span style="color:#0FCB8C;font-weight:600">▭</span> GT bbox &nbsp;'
        f'<span style="color:#F0529C;font-weight:600">●</span> GT click point</div>'
        f'</div></div>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n', type=int, default=20)
    ap.add_argument('--seed', type=int, default=1)
    args = ap.parse_args()

    d = json.load(open(DATA))['data']
    print(f'GroundCUA rows: {len(d):,}', flush=True)
    random.Random(args.seed).shuffle(d)
    picks, seen = [], set()
    for r in d:  # one per app: the pool is dominated by RStudio/Frappe Books
        app = r['image'].split('/images/')[1].split('/')[0]
        if app in seen or not os.path.isfile(r['image']):
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

    intro = (
        '<div class="dataset-intro">'
        '<div class="intro-title"><b>GroundCUA — desktop-app grounding (candidate training '
        'source, not yet used)</b></div>'
        '<div class="intro-desc">Raw dataset records, not model predictions: green box and pink '
        'dot are both ground truth. Proposed to close the ScreenSpot-Pro gap, where every reward '
        'we tried sits within noise of base (0.1398 vs 0.1404 for GT-RLVR, 51/50 paired, p=1.00) '
        'because the pool\'s only GUI source is SeeClick (web/mobile, &le;1920&times;1080). '
        'Coordinates are 0&ndash;100 percentages for both bbox and click; 2999/3000 sampled '
        'clicks land inside their own bbox.</div>'
        '<div class="intro-meta">3,131,480 records · 51,175 screenshots · 86 desktop apps · '
        f'median 1920&times;1009 · target median 0.123% of image area · showing {len(cards)} '
        '(one per app)</div></div>')
    panel = (f'<div id="p_3_groundcua" class="panel" data-cat="cat3">\n{intro}\n'
             f'{"".join(cards)}\n</div>\n')
    tab = '<button class="ds-tab" data-panel="p_3_groundcua">GroundCUA</button>'

    snip = os.path.join(REPO, '.snippets')
    open(os.path.join(snip, 'cat3_groundcua_panel.html'), 'w').write(panel)
    open(os.path.join(snip, 'cat3_groundcua_tab.html'), 'w').write(tab)

    # keep the snippet-driven tab list in sync so a future generate.py run keeps the tab
    tabs_p = os.path.join(snip, 'cat3_dataset_tabs.html')
    tabs = open(tabs_p).read()
    if 'p_3_groundcua' not in tabs:
        tabs = tabs.replace('</div>', tab + '</div>', 1)
        open(tabs_p, 'w').write(tabs)
    panels_p = os.path.join(snip, 'cat3_dataset_panels.html')
    panels = open(panels_p).read()
    if 'p_3_groundcua' not in panels:
        open(panels_p, 'w').write(panels + '\n' + panel)

    # patch the committed index.html directly: generate.py needs the full source corpora,
    # so regenerating just to add one tab is not worth it
    idx_p = os.path.join(REPO, 'index.html')
    idx = open(idx_p).read()
    if 'p_3_groundcua' in idx:
        idx = re.sub(r'<button class="ds-tab" data-panel="p_3_groundcua">.*?</button>', '', idx)
        idx = re.sub(r'<div id="p_3_groundcua" class="panel".*?\n</div>\n', '', idx, flags=re.S)
    anchor = '<button class="ds-tab" data-panel="p_3_screenspot">'
    assert anchor in idx, 'cat3 tab row not found in index.html'
    end = idx.index('</button>', idx.index(anchor)) + len('</button>')
    idx = idx[:end] + tab + idx[end:]
    m = re.search(r'<div id="p_3_screenspot" class="panel"[^>]*>', idx)
    assert m, 'ScreenSpot-Pro panel not found in index.html'
    nxt = idx.find('<div id="p_3_', m.end())
    if nxt == -1:
        nxt = idx.find('</body>', m.end())
    idx = idx[:nxt] + panel + idx[nxt:]
    open(idx_p, 'w').write(idx)
    print(f'patched index.html ({os.path.getsize(idx_p)/1e6:.1f} MB), tab under cat3')


if __name__ == '__main__':
    main()
