"""Build .snippets/cat3_v43_rm_{tab,panel}.html for the SoM RM data (cat4) tab:
v43 placement-tier pairs (SAM2-verified), 4 samples per pool, chosen|rejected
side-by-side with the exact response text the RM scores.

    python3 scripts/build_v43_rm_snippets.py
"""
import base64
import glob
import html as _html
import io
import json
import random

from PIL import Image

G = '/weka/oe-training-default/zixianm/yinuoy/grounding_rm/data'
VIZ = '/weka/oe-training-default/zixianm/yinuoy/grounding-viz'
PER_POOL = 4
MAX_DIM = 520

POOLS = [
    ('v43_near_hard', 'near_hard — m marks offset 0.45-0.85 x GT spacing (boundary graze; the placement axis v42 never learned)'),
    ('v43_near_mid', 'near_mid — 0.85-1.6 x GT spacing (clear miss, same region)'),
    ('v43_near_far', 'near_far — 1.6-3.0 x GT spacing (wrong area)'),
    ('v43_extra1_near', 'extra1_near — k correct marks + 1 wrong mark just off a GT point (hard over-mark; fewer-wins direction)'),
]


def b64(path):
    im = Image.open(path).convert('RGB')
    im.thumbnail((MAX_DIM, MAX_DIM))
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=70)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    rng = random.Random(11)
    byv = {}
    for f in sorted(glob.glob(f'{G}/som_synth_v43/train.verified.jsonl*')):
        for line in open(f):
            r = json.loads(line)
            byv.setdefault(r['meta']['pool'], []).append(r)

    parts = ['<div id="p_3_v43_rm" class="panel" data-cat="cat4">',
             '<div class="dataset-intro"><div class="intro-title"><b>v43 RM training pairs — '
             'placement tiers, SAM2-verified</b></div><div class="intro-desc">Same som_marks_v2 '
             'renders + numbered-text contract as v42; new REJECTED constructions differ from '
             'CHOSEN only in mark placement (equal count) or add one near-miss extra mark. '
             'Offsets are relative to each image\'s min GT spacing; every pair passed SAM2 '
             'point-prompted mask verification that wrong marks land off-object (geometric rule '
             'alone leaked 42-45%). Assembled v43 training set: 277,074 pairs, '
             'fewer-wins = more-wins = 39,299 (exact 1:1; v42 was 2:1).</div></div>']
    total = sum(len(v) for v in byv.values())
    parts.append(f'<h2>v43 placement pairs — {total:,} SAM2-verified rows in pool</h2>')
    for pool, desc in POOLS:
        rows = byv.get(pool, [])
        parts.append(f'<h3>{pool} ({len(rows):,} rows) — {_html.escape(desc)}</h3>')
        for r in rng.sample(rows, min(PER_POOL, len(rows))):
            ci = 0 if r['label'] == 0 else 1
            m = r['meta']
            try:
                bc, br = b64(r['images'][ci]), b64(r['images'][1 - ci])
            except Exception:
                continue
            cap = (f"phrase: <b>{_html.escape(str(m.get('phrase'))[:110])}</b> | "
                   f"k={m.get('pts_chosen')} → k_rej={m.get('pts_rejected')} | margin={m.get('margin')}")
            resp_c = _html.escape(r['responses'][ci])
            resp_r = _html.escape(r['responses'][1 - ci])
            parts.append(
                '<div style="border:1px solid #ccc;border-radius:6px;padding:8px;margin:10px 0">'
                f'<div style="font-size:12px;margin-bottom:5px">{cap}</div>'
                '<div style="display:flex;gap:8px">'
                f'<div style="flex:1;min-width:0"><img style="max-width:100%;border:3px solid #2a9d3a" '
                f'src="data:image/jpeg;base64,{bc}">'
                f'<div style="font-size:11px;color:#080"><b>CHOSEN</b></div>'
                f'<div style="font-size:11px;background:#f7f7f7;border-radius:4px;padding:4px 6px;'
                f'margin-top:3px"><b>response:</b> {resp_c}</div></div>'
                f'<div style="flex:1;min-width:0"><img style="max-width:100%;border:3px solid #d33" '
                f'src="data:image/jpeg;base64,{br}">'
                f'<div style="font-size:11px;color:#a00"><b>REJECTED</b></div>'
                f'<div style="font-size:11px;background:#f7f7f7;border-radius:4px;padding:4px 6px;'
                f'margin-top:3px"><b>response:</b> {resp_r}</div></div>'
                '</div></div>')
    parts.append('</div>')

    with open(f'{VIZ}/.snippets/cat3_v43_rm_panel.html', 'w') as f:
        f.write('\n'.join(parts))
    with open(f'{VIZ}/.snippets/cat3_v43_rm_tab.html', 'w') as f:
        f.write('<button class="ds-tab" data-panel="p_3_v43_rm">'
                'v43 RM pairs (277k, SAM2-verified placement)</button>')
    print(f'snippets written: panel {sum(len(p) for p in parts)/1e6:.1f} MB, pools {[(k, len(v)) for k, v in byv.items()]}')


if __name__ == '__main__':
    main()
