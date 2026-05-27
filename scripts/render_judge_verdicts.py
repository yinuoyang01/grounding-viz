"""Render judge-verdict QA panel: per verdict class, sample N cases with
image + yellow GT + green consensus + verdict + reason for eyeball QA.

Inputs:
  judge_candidates_v36.jsonl  (image/GT/consensus geom)
  judge_v36_full9270.jsonl    (verdict + reason)
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
VERDICT_ORDER = ['consensus_only', 'both', 'gt_only', 'neither', 'unclear']
VERDICT_COLORS = {
    'consensus_only': '#0FCB8C',  # green = upgrade
    'both':           '#0FCB8C',  # green
    'gt_only':        '#0A3235',  # dark = unchanged
    'neither':        '#F0529C',  # pink = drop
    'unclear':        '#F0529C',  # pink = drop
}


def render_card(c, verdict_info):
    from PIL import Image, ImageDraw
    ds = c['ds']
    tar_name = c.get('tar') or c['item'].split('/')[1]
    key_in_tar = c.get('key_in_tar') or c['item'].split('/', 2)[2]
    try:
        tp = resolve_tar(DS_TO_TAR_DS.get(ds, ds), tar_name)
        if not tp: return None
        ib = extract_for_ds(DS_TO_TAR_DS.get(ds, ds), tp, key_in_tar)
        if not ib: return None
        im = Image.open(io.BytesIO(ib)).convert('RGB')
    except Exception:
        return None
    W, H = im.size

    sc = 540 / max(W, H)
    nW, nH = max(1, int(W * sc)), max(1, int(H * sc))
    clean = im.resize((nW, nH))

    base = im.convert('RGBA')
    overlay = Image.new('RGBA', (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(overlay)
    lw = max(3, W // 280)

    for typ, geom in c['gt_shapes']:
        if typ == 'bbox':
            xs = [geom[0], geom[2]]; ys = [geom[1], geom[3]]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            d.rectangle([x0, y0, x1, y1], fill=(255, 210, 0, 70), outline=(255, 210, 0, 255), width=lw)
        else:
            rr = max(10, W // 100)
            d.ellipse([geom[0] - rr, geom[1] - rr, geom[0] + rr, geom[1] + rr],
                      fill=(255, 210, 0, 70), outline=(255, 210, 0, 255), width=lw)

    cx, cy = c['consensus_center']
    rr = max(14, W // 80)
    d.ellipse([cx - rr, cy - rr, cx + rr, cy + rr],
              fill=(0, 200, 80, 50), outline=(0, 200, 80, 255), width=lw + 1)

    merged = Image.alpha_composite(base, overlay).convert('RGB').resize((nW, nH))

    def _b64(im):
        b = io.BytesIO(); im.save(b, 'JPEG', quality=83)
        return base64.b64encode(b.getvalue()).decode()
    bc, bm = _b64(clean), _b64(merged)

    v = verdict_info['verdict']
    vcolor = VERDICT_COLORS.get(v, '#666')
    gtm = verdict_info.get('gt_match', '?')
    csm = verdict_info.get('consensus_match', '?')
    reason = verdict_info.get('reason', '')[:300]

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
            f'<div style="font-size:13px;line-height:1.6">'
            f'<span style="background:{vcolor};color:#FAF2E9;padding:3px 9px;border-radius:4px;'
            f'font-weight:700">{_html.escape(v)}</span><br/>'
            f'<span style="background:#FFCC00;color:#0A3235;padding:1px 6px;border-radius:3px;'
            f'font-weight:600;font-size:11px">GT={gtm}</span> &nbsp; '
            f'<span style="background:#0FCB8C;color:#FAF2E9;padding:1px 6px;border-radius:3px;'
            f'font-weight:600;font-size:11px">CONS={csm}</span>'
            f'</div>'
            f'<div style="font-size:11px;color:rgba(10,50,53,0.7);margin-top:10px;font-style:italic">'
            f'{_html.escape(reason)}</div>'
            f'</div></div>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--candidates', default='/weka/oe-training-default/oe-encoder/judge_candidates_v36.jsonl')
    ap.add_argument('--verdicts', default='/weka/oe-training-default/oe-encoder/judge_v36_full9270.jsonl')
    ap.add_argument('--per_class', type=int, default=20)
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--suffix', default='_v36')
    args = ap.parse_args()

    cands = {(c['item'], c['phrase']): c for c in (json.loads(l) for l in open(args.candidates))}
    print(f'candidates: {len(cands)}', flush=True)

    by_verdict = {v: [] for v in VERDICT_ORDER}
    for line in open(args.verdicts):
        r = json.loads(line)
        v = r['verdict']
        if v not in by_verdict: continue
        c = cands.get((r['item'], r['phrase']))
        if not c: continue
        by_verdict[v].append((c, r))
    print('per-verdict pool:', {k: len(v) for k, v in by_verdict.items()})

    rng = random.Random(args.seed)
    sections = []
    total_cards = total_fails = 0
    for v in VERDICT_ORDER:
        pool = by_verdict[v]
        picks = rng.sample(pool, min(args.per_class, len(pool)))
        cards = []
        fails = 0
        for c, ri in picks:
            card = render_card(c, ri)
            if card: cards.append(card)
            else: fails += 1
        total_cards += len(cards); total_fails += fails
        header = (f'<h2 style="color:{VERDICT_COLORS.get(v, "#0A3235")};'
                  f'margin-top:24px;padding:8px 12px;border-left:6px solid {VERDICT_COLORS.get(v, "#0A3235")};'
                  f'background:rgba(10,50,53,0.04)">'
                  f'{_html.escape(v)} &mdash; {len(cards)} samples '
                  f'(pool {len(pool)})</h2>')
        sections.append(header + '\n' + '\n'.join(cards))

    panel_id = f'p_3_judge_verdicts{args.suffix}'
    intro = (
        '<div class="dataset-intro">'
        f'<div class="intro-title"><b>Judge verdicts v36 &mdash; quality QA</b></div>'
        f'<div class="intro-desc">{total_cards} samples across 5 verdict classes '
        f'(~{args.per_class}/class). For each: image + <b style="color:#B58800">yellow GT</b> + '
        f'<b style="color:#0A8A4E">green consensus</b> + GPT-5 verdict + its reasoning. '
        f'Use this to spot-check whether the verdict matches the visual evidence.</div>'
        f'<div class="intro-meta">{total_fails} render failures</div></div>')

    panel = (f'<div id="{panel_id}" class="panel" data-cat="cat3">\n{intro}\n'
             + '\n'.join(sections) + '\n</div>\n')
    tab = f'<button class="ds-tab" data-panel="{panel_id}">Judge verdicts v36 (QA)</button>'

    os.makedirs(SNIP, exist_ok=True)
    tab_name = f'cat3_judge_verdicts{args.suffix}_tab.html'
    panel_name = f'cat3_judge_verdicts{args.suffix}_panel.html'
    with open(os.path.join(SNIP, tab_name), 'w') as f: f.write(tab)
    with open(os.path.join(SNIP, panel_name), 'w') as f: f.write(panel)
    print(f'wrote {tab_name} + {panel_name} ({len(panel):,}b, {total_cards} cards)')


if __name__ == '__main__':
    main()
