"""Build .snippets/cat3_v44_arenas_{tab,panel}.html for the SoM RM data (cat4) tab:
the seven v44 rollout-arena datasets (V44_ARENAS.md), 3 samples each with GT drawn
on the image (points = filled circles + index, bbox = rectangle).

    python3 scripts/build_v44_arenas_snippets.py
"""
import base64
import html as _html
import io
import json
import mmap
import random

from PIL import Image, ImageDraw

VIZ = '/weka/oe-training-default/zixianm/yinuoy/grounding-viz'
PIXMO = '/weka/oe-training-default/mm-olmo/torch_datasets/pixmo_datasets'
WEB = '/weka/oe-training-default/webolmo/datasets'
GUISYN_PARQUET = ('/weka/oe-training-default/zixianm/yinuoy/grounding_rm/rl/'
                  'data_guisyn/grounding_rl_train.parquet')
MAX_DIM = 480
SEED = 7
N_SAMPLES = 3


def to_b64(im):
    buf = io.BytesIO()
    im.convert('RGB').save(buf, 'JPEG', quality=70)
    return base64.b64encode(buf.getvalue()).decode()


def prep(im):
    """-> (resized copy max 480, scale factor from original px)."""
    im = im.convert('RGB')
    w, h = im.size
    s = min(1.0, MAX_DIM / max(w, h))
    if s < 1.0:
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.LANCZOS)
    return im, s


def draw_points(im, pts_px, scale):
    """pts_px in ORIGINAL pixel coords; draws filled circles + index on resized im."""
    d = ImageDraw.Draw(im)
    r = 6
    for i, (x, y) in enumerate(pts_px):
        x, y = x * scale, y * scale
        d.ellipse([x - r, y - r, x + r, y + r], fill='#f0529c', outline='white', width=2)
        d.text((x + r + 1, y - r - 1), str(i + 1), fill='#f0529c',
               stroke_width=2, stroke_fill='white')
    return im


def draw_bbox(im, box_px, scale):
    """box_px = (x1,y1,x2,y2) in ORIGINAL pixel coords."""
    d = ImageDraw.Draw(im)
    x1, y1, x2, y2 = [v * scale for v in box_px]
    d.rectangle([x1, y1, x2, y2], outline='#2a9d3a', width=3)
    return im


def card(im_b64, caption):
    return ('<div style="border:1px solid #ccc;border-radius:6px;padding:8px;'
            'margin:10px 0;max-width:520px">'
            f'<img style="max-width:100%" src="data:image/jpeg;base64,{im_b64}">'
            f'<div style="font-size:12px;margin-top:5px">{caption}</div></div>')


# ---------------------------------------------------------------- pixmo arenas

def pixmo_points_cards(dirname, rng):
    import datasets
    ds = datasets.load_from_disk(f'{PIXMO}/{dirname}')['train']
    cards = []
    for idx in rng.sample(range(len(ds)), 40):
        if len(cards) >= N_SAMPLES:
            break
        r = ds[idx]
        # pick the label with the most GT points that has >=1
        li = max(range(len(r['label'])), key=lambda i: r['count'][i])
        if r['count'][li] < 1:
            continue
        try:
            im = Image.open(r['image'])
        except Exception:
            continue
        W, H = im.size
        pts = [(p['x'] / 100 * W, p['y'] / 100 * H) for p in r['points'][li]]
        rim, s = prep(im)
        draw_points(rim, pts, s)
        cap = (f'prompt: <b>point to all {_html.escape(r["label"][li])}</b> — '
               f'GT: {r["count"][li]} point(s), x/y in 0-100 of image size')
        cards.append(card(to_b64(rim), cap))
    return cards


def pixmo_count_cards(rng):
    import datasets
    ds = datasets.load_from_disk(f'{PIXMO}/count')['train']
    cards = []
    want_zero = True   # show one count=0 abstain row among the 3
    for idx in rng.sample(range(len(ds)), 60):
        if len(cards) >= N_SAMPLES:
            break
        r = ds[idx]
        if r['count'] == 0 and not want_zero:
            continue
        if r['count'] == 0:
            want_zero = False
        try:
            im = Image.open(r['image'])
        except Exception:
            continue
        W, H = im.size
        pts = [(x / 100 * W, y / 100 * H)
               for x, y in zip(r['points']['x'], r['points']['y'])]
        rim, s = prep(im)
        draw_points(rim, pts, s)
        cap = (f'prompt: <b>count the {_html.escape(r["label"])}</b> — '
               f'<b>GT count = {r["count"]}</b>'
               + ('' if r['count'] else ' (explicit absent-object row, no GT points)'))
        cards.append(card(to_b64(rim), cap))
    return cards


def cosyn_point_cards(rng):
    import datasets
    ds = datasets.load_from_disk(f'{PIXMO}/cosyn-point')['train']
    cards = []
    for idx in rng.sample(range(len(ds)), 40):
        if len(cards) >= N_SAMPLES:
            break
        r = ds[idx]
        qi = rng.randrange(len(r['questions']))
        ap = r['answer_points'][qi]
        if not ap['x']:
            continue
        try:
            im = Image.open(r['image'])
        except Exception:
            continue
        W, H = im.size
        pts = [(x / 100 * W, y / 100 * H) for x, y in zip(ap['x'], ap['y'])]
        rim, s = prep(im)
        draw_points(rim, pts, s)
        cap = (f'question: <b>{_html.escape(r["questions"][qi])}</b> — '
               f'target: {_html.escape(r["names"][qi])}, {len(pts)} GT point(s) in 0-100')
        cards.append(card(to_b64(rim), cap))
    return cards


# ---------------------------------------------------------------- GUI arenas

def guisyn_cards(rng):
    import pandas as pd
    df = pd.read_parquet(GUISYN_PARQUET)
    cards = []
    for subset in ('guisyn_desktop', 'guisyn_mobile', 'guisyn_web'):
        sub = df[df['source_dataset'] == subset]
        for idx in rng.sample(range(len(sub)), 8):
            r = sub.iloc[idx]
            try:
                im = Image.open(r['image'])
            except Exception:
                continue
            W, H = im.size
            gt = json.loads(r['reward_model']['ground_truth'])
            pts = [(gx / 999 * W, gy / 999 * H) for gx, gy in gt]   # 0-999 centroid space
            rim, s = prep(im)
            draw_points(rim, pts, s)
            q = r['prompt'][0]['content'].replace('<image>', '')
            cap = (f'[{subset.split("_")[1]}] prompt: <b>{_html.escape(q)}</b> — '
                   'GT = element centroid (0-999 coords; judge point-in-bbox)')
            cards.append(card(to_b64(rim), cap))
            break
    return cards


def synthetic_ground_cards(rng):
    dec = json.JSONDecoder()
    gpt5_path = f'{WEB}/webolmo_synthetic_ground/gpt5_outputs_all_processed.json'
    f5 = open(gpt5_path, 'rb')
    mm = mmap.mmap(f5.fileno(), 0, access=mmap.ACCESS_READ)

    def gpt5_for(key):
        pos = mm.find(f'"{key}":'.encode())
        if pos < 0:
            return None
        chunk = mm[pos:pos + 2_000_000].decode('utf-8', 'ignore')
        obj, _ = dec.raw_decode(chunk[chunk.index(':') + 1:].lstrip())
        return obj

    cards = []
    for website in ('amazon', 'allrecipes', 'wikipedia', 'github'):
        try:
            recs = json.load(open(f'{WEB}/webolmo_synthetic_ground/train_{website}.json'))
        except Exception:
            continue
        for rec in rng.sample(recs, min(30, len(recs))):
            els = [e for e in rec['elements'] if e.get('clickable')]
            if not els:
                continue
            key = f"{rec['website']}__{rec['traj_id']}__{rec['step_id']}"
            g = gpt5_for(key)
            if not g:
                continue
            els = [e for e in els if e['bid'] in g]
            if not els:
                continue
            e = rng.choice(els)
            try:
                im = Image.open(rec['image_path'])
            except Exception:
                continue
            x, y, w, h = e['bbox']            # pixels, image size == viewport
            rim, s = prep(im)
            draw_bbox(rim, (x, y, x + w, y + h), s)
            cap = (f'[{website}] GPT-5 query: <b>{_html.escape(g[e["bid"]]["query"])}</b> — '
                   f'GT = element bbox (px), element: {_html.escape(e["name"][:80])}')
            cards.append(card(to_b64(rim), cap))
            break
        if len(cards) >= N_SAMPLES:
            break
    mm.close(); f5.close()
    return cards


def ground_cua_cards(rng):
    path = f'{WEB}/GroundCUA/formatted_data.json'
    dec = json.JSONDecoder()
    f = open(path, 'rb')
    mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
    size = mm.size()
    cards, seen_imgs = [], set()
    for off in sorted(rng.sample(range(0, size - 5_000_000, 4096), 20)):
        if len(cards) >= N_SAMPLES:
            break
        pos = mm.find(b'{"image":', off)
        if pos < 0:
            continue
        chunk = mm[pos:pos + 3_000_000].decode('utf-8', 'ignore')
        try:
            r, _ = dec.raw_decode(chunk)
        except Exception:
            continue
        if r['image'] in seen_imgs:
            continue
        try:
            im = Image.open(r['image'])
        except Exception:
            continue
        seen_imgs.add(r['image'])
        W, H = im.size
        x1, y1, x2, y2 = r['metadata']['bbox']   # 0-100 normalized
        rim, s = prep(im)
        ans = json.loads(r['answer'])
        draw_points(rim, [(ans['x'] / 100 * W, ans['y'] / 100 * H)], s)
        # bbox second: tiny GUI elements would be fully hidden under the click dot
        draw_bbox(rim, (x1 / 100 * W, y1 / 100 * H, x2 / 100 * W, y2 / 100 * H), s)
        app = r['image'].split('/images/')[1].split('/')[0]
        cap = (f'[{_html.escape(app)}] question: <b>{_html.escape(r["question"][:140])}</b> — '
               'GT = element bbox (green, 0-100 coords); pink dot = pre-sampled click '
               '(gaussian in bbox; judge uses the bbox)')
        cards.append(card(to_b64(rim), cap))
    mm.close(); f.close()
    return cards


# ---------------------------------------------------------------- assembly

ARENAS = [
    ('pixmo_points_train', 'pixmo_datasets/points-pointing — 152,858 train images',
     'GT = per-label point lists (x,y in 0-100); prompt "point to all &lt;label&gt;". '
     'Core natural-image pointing, matches the GRPO distribution.', pixmo_points_cards, 'points-pointing'),
    ('pixmo_points_high_freq', 'pixmo_datasets/points-counting — 70,714 train images',
     'GT = dense per-label point lists + count (0-100 coords); the "counting" '
     'collection_method subset (naming inverted). Multi-point regime, ~10+ instances/label.',
     pixmo_points_cards, 'points-counting'),
    ('pixmo_count', 'pixmo_datasets/count — 36,916 train rows',
     'GT = single label + integer count (+ points when count&gt;0, 0-100 coords); count=0 '
     'rows are free abstain prompts.', pixmo_count_cards, None),
    ('cosyn_point', 'pixmo_datasets/cosyn-point — 68,051 train images (~5 questions each)',
     'Synthetic rendered docs/UIs; GT = per-question answer point(s) in 0-100, exact by '
     'construction (renderer).', cosyn_point_cards, None),
    ('molmo2_syn_point (GUISyn)', 'allenai/MolmoPoint-GUISyn — 36,192 HF train images; '
     'extracted RL prompts: 144,290 element-intent rows',
     'GT = element bbox (center+w/h, px); parquet ground_truth = centroid in 0-999; judge '
     'point-in-bbox. Subsets desktop/mobile/web.', guisyn_cards, None),
    ('synthetic_ground_point', 'webolmo_synthetic_ground — 466,599 train screenshots / '
     '4.1M clickable elements, 23 websites',
     'GT = element bbox [x,y,w,h] in px (image == viewport); prompt = GPT-5 query per '
     'element; judge point-in-bbox.', synthetic_ground_cards, None),
    ('ground_cua', 'GroundCUA/formatted_data.json — 3,131,480 records / 50,492 unique '
     'screenshots (desktop apps)',
     'GT = element bbox in 0-100 (metadata.bbox); the stored click answer is a gaussian '
     'sample inside it — judge point-in-bbox, not the click.', ground_cua_cards, None),
]


def main():
    parts = ['<div id="p_3_v44_arenas" class="panel" data-cat="cat4">',
             '<div class="dataset-intro"><div class="intro-title"><b>v44 rollout arenas — '
             'the 7 datasets we roll Molmo2-4B out on</b></div><div class="intro-desc">'
             'v44 pairs come from REAL rollouts judged against these datasets\' GT (we do '
             'not reuse their synthetic point strings). First batch: 16,000 prompts x 8 '
             'samples = 128k rollouts; max 1 prompt/image, train splits only, '
             'banned_images_v44 filter. GT overlays below: pink filled circles = GT points '
             '(numbered), green rectangle = GT element bbox.</div></div>']
    for name, count_line, desc, fn, arg in ARENAS:
        rng = random.Random(SEED)
        print(f'[{name}] ...', flush=True)
        cards = fn(arg, rng) if arg else fn(rng)
        parts.append(f'<h3>{_html.escape(name)} — {count_line}</h3>'
                     f'<div style="font-size:13px;color:#444;margin:2px 0 6px">{desc}</div>')
        parts.extend(cards)
        print(f'[{name}] {len(cards)} cards', flush=True)
    parts.append('</div>')

    with open(f'{VIZ}/.snippets/cat3_v44_arenas_panel.html', 'w') as f:
        f.write('\n'.join(parts))
    with open(f'{VIZ}/.snippets/cat3_v44_arenas_tab.html', 'w') as f:
        f.write('<button class="ds-tab" data-panel="p_3_v44_arenas">'
                'v44 arenas (7 datasets, real-rollout考场)</button>')
    print(f'panel size {sum(len(p) for p in parts) / 1e6:.1f} MB')


if __name__ == '__main__':
    main()
