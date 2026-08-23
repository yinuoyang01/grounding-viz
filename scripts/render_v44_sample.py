"""Render v44 SAMPLE pairs (real GRPO-rollout rejected sides) to v44_sample.html.

v44 recipe: chosen = SoM render of the TRUE GT points; rejected = render of a REAL
policy rollout the GT rule judged wrong (flavor gt_vs_rollout), plus real-vs-real
pairs (correct rollout vs wrong rollout on the same prompt). Same som_marks_v2
renders and v42-numbered text contract as v42/v43; only the rejected-side SOURCE is
new: the policy's own mistakes instead of synthetic offsets.

    python3 scripts/render_v44_sample.py --per_flavor 7
"""
import argparse
import base64
import html as _html
import io
import json
import random

from PIL import Image

SRC = '/weka/oe-training-default/zixianm/yinuoy/grounding_rm/data/som_v44_sample'
OUT = '/weka/oe-training-default/zixianm/yinuoy/grounding-viz/v44_sample.html'

DESC = {
    'gt_vs_rollout': 'gt_vs_rollout — chosen = TRUE GT points; rejected = a real rollout judged WRONG',
    'real_vs_real': 'real_vs_real — chosen = a CORRECT rollout; rejected = a WRONG rollout, same prompt',
}


def b64(im, max_dim, q=72):
    im = im.copy()
    im.thumbnail((max_dim, max_dim))
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=q)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per_flavor', type=int, default=7)
    ap.add_argument('--seed', type=int, default=44)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    byf = {}
    for line in open(f'{SRC}/sample.jsonl'):
        r = json.loads(line)
        byf.setdefault(r['meta']['flavor'], []).append(r)
    stats = json.load(open(f'{SRC}/stats.json'))

    cards = []
    for flavor in ('gt_vs_rollout', 'real_vs_real'):
        rows = byf.get(flavor, [])
        cards.append(f'<h2>{_html.escape(DESC[flavor])} <span style="font-weight:normal;'
                     f'font-size:14px;color:#666">({len(rows)} pairs in sample)</span></h2>')
        for r in rng.sample(rows, min(args.per_flavor, len(rows))):
            ci = r['label']
            m = r['meta']
            try:
                imc = Image.open(r['images'][ci]).convert('RGB')
                imr = Image.open(r['images'][1 - ci]).convert('RGB')
            except Exception:
                continue
            title = (f"“{_html.escape(str(m.get('phrase'))[:110])}” "
                     f"| {m['source_dataset']} | k={m['k']} vs k_rej={m['k_rej']} "
                     f"| direction={m['direction']}")
            verdict = (f"judge: {_html.escape(m['judge'])} — f1_chosen={m['f1_chosen']}, "
                       f"f1_rejected={m['f1_rejected']}")
            origin = f"rejected from arm <b>{m['arm']}</b> @ step {m['step']}"
            raw = _html.escape((m.get('rejected_output') or '')[:220])
            cho_lbl = ('marks on TRUE GT points' if flavor == 'gt_vs_rollout'
                       else 'real rollout judged CORRECT (f1=1)')
            cards.append(
                '<div style="border:1px solid #ccc;border-radius:8px;padding:10px;margin:14px 0">'
                f'<div style="font-size:13px;margin-bottom:2px">{title}</div>'
                f'<div style="font-size:12px;color:#555;margin-bottom:6px">{verdict} · {origin}</div>'
                '<div style="display:flex;gap:8px">'
                f'<div style="flex:1"><img style="width:100%" src="data:image/jpeg;base64,{b64(imc, 820)}">'
                f'<div style="font-size:12px;color:#080"><b>CHOSEN</b> — {cho_lbl}</div></div>'
                f'<div style="flex:1"><img style="width:100%" src="data:image/jpeg;base64,{b64(imr, 820)}">'
                f'<div style="font-size:12px;color:#a00"><b>REJECTED</b> — real rollout judged WRONG</div>'
                f'<div style="font-size:11px;color:#888;word-break:break-all">{raw}</div></div></div>'
                '</div>')

    dir_rows = ''.join(
        f'<tr><td>{fl}</td>' + ''.join(
            f'<td>{stats["direction"].get(fl, {}).get(d, 0)}</td>'
            for d in ('fewer', 'more', 'equal', 'none_wins', 'marked_wins'))
        + f'<td><b>{stats["by_flavor"].get(fl, 0)}</b></td></tr>'
        for fl in ('gt_vs_rollout', 'real_vs_real'))
    page = ('<!DOCTYPE html><html><head><meta charset="utf-8">'
            '<title>v44 sample — rollout-harvested RM pairs</title></head>'
            '<body style="font-family:sans-serif;max-width:1200px;margin:0 auto;padding:20px">'
            '<h1>v44 sample — RM pairs harvested from real GRPO rollouts</h1>'
            '<p>Rejected sides are REAL Molmo2-4B rollouts from the four trimix_molmo_v2 GRPO '
            'arms (som_rm / hybrid / gt_dist / hybrid_w0.25), judged against the parquet GT '
            'centroids with a greedy-match rule (tol=50/1000; wrong iff F1&le;0.5; zero points '
            'with non-empty GT = wrong; 0.5&lt;F1&lt;1 excluded as ambiguous). The dump score '
            'field is each arm\'s training reward, not GT correctness, so it was not used. '
            'Same som_marks_v2 renders + v42-numbered text as v42/v43. Direction distribution '
            'is measured, NOT rebalanced.</p>'
            '<table border="1" cellpadding="6" style="border-collapse:collapse;font-size:13px">'
            '<tr><th>flavor</th><th>fewer wins</th><th>more wins</th><th>equal</th>'
            '<th>none wins</th><th>marked wins</th><th>total</th></tr>'
            + dir_rows + '</table>'
            + '\n'.join(cards) + '</body></html>')
    with open(OUT, 'w') as f:
        f.write(page)
    print(f'{OUT}: {len(cards)} blocks, {len(page)/1e6:.1f} MB')


if __name__ == '__main__':
    main()
