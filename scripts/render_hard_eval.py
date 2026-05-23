"""Render extra cat3 dataset tabs for the HARD eval benchmarks (V*Bench,
ScreenSpot-Pro). Reads Molmo2's eval outputs (point-in-bbox scoring) and
appends a tab+panel for each to cat3_dataset_tabs.html / cat3_dataset_panels.html
(after render_cascade_judge.py has already written them).

Each card shows: left = original image · right = same image with yellow GT bbox
+ red ring at Molmo2's predicted point. Mix of hits and misses per dataset.
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
from eval_vstar import load_vstar  # noqa: E402
from eval_screenspot import load_screenspot  # noqa: E402

SNIP = '/weka/oe-training-default/zixianm/yinuoy/grounding-viz/.snippets'
V_OUT = '/weka/oe-training-default/oe-encoder/vstar_eval/molmo2_vstar.jsonl'
S_OUT = '/weka/oe-training-default/oe-encoder/screenspot_eval/molmo2_screenspot_pro.jsonl'
V_ROOT = '/weka/oe-training-default/jamesp/data/vstar_bench'
S_ROOT = '/weka/oe-training-default/webolmo/datasets/screenspot_pro'


def render_card(rec, row, max_side=480):
    from PIL import Image, ImageDraw
    try:
        im = Image.open(rec['img_path']).convert('RGB')
        W, H = im.size
    except Exception:
        return None
    sc = max_side / max(W, H)
    nW, nH = max(1, int(W * sc)), max(1, int(H * sc))
    clean = im.resize((nW, nH))
    marked = clean.copy()
    d = ImageDraw.Draw(marked)
    # GT bbox (yellow). V* uses gt_xyxy list of boxes; SS uses single bbox.
    boxes = rec.get('gt_xyxy') or [rec['bbox']]
    lw = max(2, nW // 220)
    for bb in boxes:
        x1, y1, x2, y2 = bb
        d.rectangle([x1 * sc, y1 * sc, x2 * sc, y2 * sc],
                    outline=(255, 210, 0), width=lw)
    # Molmo2 pred rings (red)
    rad = max(6, nW // 90)
    for (x, y) in (row.get('pred') or []):
        cx, cy = x * sc, y * sc
        d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=(255, 255, 255), width=3)
        d.ellipse([cx - rad, cy - rad, cx + rad, cy + rad], outline=(220, 30, 60), width=3)

    def _b64(im):
        b = io.BytesIO(); im.save(b, 'JPEG', quality=82)
        return base64.b64encode(b.getvalue()).decode()
    bc, bm = _b64(clean), _b64(marked)
    hit = row.get('hit')
    tag_bg = '#0FCB8C' if hit else '#F0529C'
    tag = 'HIT' if hit else 'MISS'
    extra = (f"ui={row.get('ui_type')} · group={row.get('group')}" if 'ui_type' in row
             else f"sub={row.get('sub')}")
    ph = _html.escape((row.get('phrase') or '')[:280])
    ST = ('display:grid;grid-template-columns:720px 1fr;gap:16px;padding:12px;margin:8px 0;'
          'background:var(--bg,#FAF2E9);border:1px solid rgba(10,50,53,0.15);border-radius:8px')
    chip = (f'<span style="background:{tag_bg};color:#FAF2E9;padding:2px 8px;'
            f'border-radius:5px;font-size:11px;font-weight:700">{tag}</span>')
    return (f'<div style="{ST}">'
            f'<div style="display:flex;gap:8px">'
            f'<img src="data:image/jpeg;base64,{bc}" style="width:50%;border-radius:4px"/>'
            f'<img src="data:image/jpeg;base64,{bm}" style="width:50%;border-radius:4px"/>'
            f'</div>'
            f'<div>'
            f'<div style="color:rgba(10,50,53,0.55);font-family:ui-monospace,Menlo,monospace;'
            f'font-size:11px;margin-bottom:6px">{_html.escape(rec["key"])} · {extra}</div>'
            f'<div style="color:#0A3235;font-size:13px;font-weight:700;margin-bottom:6px">{ph}</div>'
            f'{chip}<div style="font-size:11px;color:rgba(10,50,53,0.7);margin-top:6px">'
            f'Yellow = GT bbox · Red ring = Molmo2 prediction</div></div></div>')


def build_panel(panel_id, title, intro_desc, intro_meta, cards):
    intro = (f'<div class="dataset-intro">'
             f'<div class="intro-title"><b>{title}</b></div>'
             f'<div class="intro-desc">{intro_desc}</div>'
             f'<div class="intro-meta">{intro_meta}</div></div>')
    return (f'<div id="{panel_id}" class="panel" data-cat="cat3">\n{intro}\n'
            + '\n'.join(cards) + '\n</div>\n')


def vstar_panel(per_class=4, seed=3):
    recs = {r['key']: r for r in load_vstar(V_ROOT)}
    rows = [json.loads(l) for l in open(V_OUT) if 'error' not in l]
    hits = [r for r in rows if r.get('hit')]
    misses = [r for r in rows if not r.get('hit')]
    rng = random.Random(seed)
    sample = (rng.sample(hits, min(per_class, len(hits)))
              + rng.sample(misses, min(per_class * 3, len(misses))))
    cards = []
    for row in sample:
        rec = recs.get(row['key'])
        if not rec: continue
        c = render_card(rec, row, max_side=520)
        if c: cards.append(c)
    h = sum(r['hit'] for r in rows); n = len(rows)
    return build_panel('p_3_vstar', 'V*Bench — Molmo2 small-object grounding',
        ('High-res images (2000-4000px), GT targets 0.03-0.4% of image area. '
         'Visual-search benchmark (not pure grounding). Molmo2 miss rate is heavily '
         'resolution-bound (target sub-token-size after smart-resize), not a '
         'grounding-decision failure.'),
        (f'{n} items · Molmo2 hit {h}/{n} = {h/n:.1%} · showing {len(cards)} '
         f'(~{per_class} hits + ~{per_class*3} misses)'), cards)


def screenspot_panel(per_class=4, seed=3):
    recs = {r['key']: r for r in load_screenspot(S_ROOT)}
    rows = [json.loads(l) for l in open(S_OUT) if 'error' not in l]
    hits = [r for r in rows if r.get('hit')]
    misses = [r for r in rows if not r.get('hit')]
    rng = random.Random(seed)
    sample = (rng.sample(hits, min(per_class, len(hits)))
              + rng.sample(misses, min(per_class * 3, len(misses))))
    cards = []
    for row in sample:
        rec = recs.get(row['key'])
        if not rec: continue
        c = render_card(rec, row, max_side=520)
        if c: cards.append(c)
    h = sum(r['hit'] for r in rows); n = len(rows)
    return build_panel('p_3_screenspot', 'ScreenSpot-Pro — Molmo2 GUI grounding',
        ('Professional high-resolution (4K) software screenshots — engineering, '
         'creative, scientific tools. UI targets are tiny icons / text. Molmo2 is '
         'severely under-resolved (icons reduce to ~1 visual token after the model\'s '
         'downsampling); also an OOD domain for it.'),
        (f'{n} items · Molmo2 hit {h}/{n} = {h/n:.1%} · showing {len(cards)} '
         f'(~{per_class} hits + ~{per_class*3} misses)'), cards)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per_class', type=int, default=4)
    args = ap.parse_args()

    panels = []
    tabs = []
    for pid, title, build in [
        ('p_3_vstar', 'V*Bench', lambda: vstar_panel(args.per_class)),
        ('p_3_screenspot', 'ScreenSpot-Pro', lambda: screenspot_panel(args.per_class)),
    ]:
        print(f'building {title}...', flush=True)
        panels.append(build())
        tabs.append(f'<button class="ds-tab" data-panel="{pid}">{title}</button>')

    # APPEND tab buttons to cat3_dataset_tabs.html (before closing </div>)
    tabs_path = os.path.join(SNIP, 'cat3_dataset_tabs.html')
    with open(tabs_path) as f: cur_tabs = f.read()
    extra_tabs = ''.join(tabs)
    if extra_tabs in cur_tabs:
        new_tabs = cur_tabs  # already appended (re-run idempotent)
    else:
        new_tabs = cur_tabs.replace('</div>', extra_tabs + '</div>', 1)
    with open(tabs_path, 'w') as f: f.write(new_tabs)

    # APPEND panels to cat3_dataset_panels.html
    panels_path = os.path.join(SNIP, 'cat3_dataset_panels.html')
    with open(panels_path) as f: cur_panels = f.read()
    if 'id="p_3_vstar"' in cur_panels:
        # strip any old ones then re-append
        import re as _re
        cur_panels = _re.sub(r'<div id="p_3_vstar"[\s\S]*?(?=<div id="p_3_screenspot"|</body>|$)',
                             '', cur_panels)
        cur_panels = _re.sub(r'<div id="p_3_screenspot"[\s\S]*?(?=<div id=|</body>|$)',
                             '', cur_panels)
    with open(panels_path, 'w') as f:
        f.write(cur_panels.rstrip() + '\n' + '\n'.join(panels))
    print(f'appended 2 dataset tabs (V*Bench, ScreenSpot-Pro) to cat3 snippets')


if __name__ == '__main__':
    main()
