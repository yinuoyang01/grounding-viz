"""Render the model-spectrum harvest pool as a cat3 ("Grounding data filtering")
panel — writes .snippets/cat3_judge_tab.html + cat3_judge_panel.html, which
generate.py injects as an extra tab under that category.

Pool = harvest_aggregate.py output: per (benchmark,item,phrase) group, the
predictions of a spectrum of models (weak->strong), each labelled pos/neg vs GT.
A card shows the image + GT region + every model's predicted point
(green = correct, red = wrong).
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
from eval_pointing import LOADERS, BR  # noqa: E402

POOL = '/weka/oe-training-default/oe-encoder/pointing_eval/rm_pool_pointing.jsonl'
SNIP = '/weka/oe-training-default/zixianm/yinuoy/grounding-viz/.snippets'
SHORT = {'molmo2-4b': 'Molmo2-4B', 'molmo7b-d': 'Molmo-7B', 'molmoE-1b': 'MolmoE-1B',
         'qwen2.5vl-3b': 'Qwen2.5VL-3B', 'qwen2.5vl-7b': 'Qwen2.5VL-7B',
         'internvl3-8b': 'InternVL3-8B'}


def _img(r):
    from PIL import Image
    b = open(r['img_path'], 'rb').read() if 'img_path' in r else r['img_bytes']
    return Image.open(io.BytesIO(b)).convert('RGB')


def _mask(r):
    from PIL import Image
    b = open(r['mask_path'], 'rb').read() if 'mask_path' in r else r['mask_bytes']
    return Image.open(io.BytesIO(b)).convert('L')


def render_card(r, group):
    import numpy as np
    from PIL import Image, ImageDraw
    try:
        im = _img(r)
        W, H = im.size
        mk = _mask(r)
        if mk.size != (W, H):
            mk = mk.resize((W, H))
    except Exception:
        return None
    d = ImageDraw.Draw(im)
    m = np.array(mk)
    ys, xs = np.where(m > 10)
    if len(xs):
        lw = max(2, W // 320)
        d.rectangle([int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
                    outline=(255, 210, 0), width=lw)
    rad = max(6, W // 120)
    for p in group['preds']:
        col = (0, 200, 80) if p['label'] == 'pos' else (235, 40, 60)
        for (x, y) in (p.get('pred') or []):
            d.ellipse([x - rad, y - rad, x + rad, y + rad], outline=(255, 255, 255), width=2)
            d.ellipse([x - rad, y - rad, x + rad, y + rad], outline=col, width=max(2, rad // 3))
    sc = 520 / max(im.size)
    im = im.resize((max(1, int(im.size[0] * sc)), max(1, int(im.size[1] * sc))))
    buf = io.BytesIO(); im.save(buf, 'JPEG', quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode()

    chips = ''
    for p in group['preds']:
        ok = p['label'] == 'pos'
        bg = '#0FCB8C' if ok else '#F0529C'
        chips += (f'<span style="background:{bg};color:#FAF2E9;padding:2px 7px;border-radius:4px;'
                  f'font-size:10px;font-weight:700;margin:0 4px 4px 0;display:inline-block">'
                  f'{_html.escape(SHORT.get(p["model"], p["model"]))}: {"OK" if ok else "X"}</span>')
    ST = ('display:grid;grid-template-columns:420px 1fr;gap:16px;padding:12px;margin:8px 0;'
          'background:var(--bg,#FAF2E9);border:1px solid rgba(10,50,53,0.15);border-radius:8px')
    return (f'<div style="{ST}">'
            f'<div><img src="data:image/jpeg;base64,{b64}" style="width:100%;border-radius:4px"/></div>'
            f'<div>'
            f'<div style="color:rgba(10,50,53,0.55);font-family:ui-monospace,Menlo,monospace;'
            f'font-size:11px;margin-bottom:6px">{_html.escape(group["bench"])} / '
            f'{_html.escape(str(group.get("category","-")))}</div>'
            f'<div style="color:#0A3235;font-size:13px;font-weight:700;margin-bottom:8px">'
            f'{_html.escape((group.get("phrase") or "")[:300])}</div>'
            f'<div>{chips}</div></div></div>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per_tab', type=int, default=20)
    ap.add_argument('--seed', type=int, default=5)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    recs = {}
    for ds in LOADERS:
        for r in LOADERS[ds](f'{BR}/{ds}'):
            recs[f"{r['ds']}/{r['key']}"] = r

    pool = [json.loads(l) for l in open(POOL)]
    usable = [g for g in pool if g.get('has_pos') and g.get('has_neg')]
    print(f'pool {len(pool)} | usable {len(usable)}', flush=True)

    cards = []
    for g in rng.sample(usable, min(args.per_tab, len(usable))):
        r = recs.get(g['item'])
        if not r:
            continue
        c = render_card(r, g)
        if c:
            cards.append(c)

    n_pred = sum(len(g['preds']) for g in pool)
    n_pos = sum(sum(p['label'] == 'pos' for p in g['preds']) for g in pool)
    intro = (
        '<div class="dataset-intro">'
        '<div class="intro-title"><b>Model-spectrum harvest pool — REAL grounding errors</b></div>'
        f'<div class="intro-desc">A spectrum of 6 pointing models (weak&rarr;strong, 3 architectures: '
        f'MolmoE-1B / Molmo-7B / Molmo2-4B / Qwen2.5-VL-3B+7B / InternVL3-8B) run on the spatial-'
        f'pointing benchmarks. Each prediction scored vs GT: a HIT is a real positive, a MISS is a '
        f'REAL hard negative (an actual model error, not constructed). Grouped by (benchmark, item, '
        f'phrase) so the RM sees good-vs-bad predictions for the same query.</div>'
        f'<div class="intro-meta">{n_pred:,} predictions ({n_pos:,} correct / {n_pred-n_pos:,} wrong) '
        f'&middot; {len(pool):,} groups &middot; <b>{len(usable):,} groups with both a correct and a '
        f'wrong pred = directly usable RM preference pairs</b></div></div>')

    panel = (f'<div id="p_3_spectrum_pool" class="panel" data-cat="cat3">\n{intro}\n'
             f'<div style="font-size:11px;color:rgba(10,50,53,0.55);margin:8px 0">'
             f'Yellow box = GT region &middot; green dot = model hit &middot; red dot = model miss '
             f'&middot; showing {len(cards)} sampled pairs</div>\n'
             + '\n'.join(cards) + '\n</div>\n')
    tab = '<button class="ds-tab" data-panel="p_3_spectrum_pool">Spectrum Pool</button>'

    os.makedirs(SNIP, exist_ok=True)
    with open(os.path.join(SNIP, 'cat3_judge_tab.html'), 'w') as f:
        f.write(tab)
    with open(os.path.join(SNIP, 'cat3_judge_panel.html'), 'w') as f:
        f.write(panel)
    print(f'wrote cat3_judge_tab.html + cat3_judge_panel.html ({len(panel):,}b, {len(cards)} cards)')


if __name__ == '__main__':
    main()
