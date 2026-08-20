"""v41_data.html: browse EVERY data source feeding the v41 line.

Section 1 - RM training pairs (what the v41 RM learns from), one block per pool:
    re-rendered old pools: som_v2 (118k), som_v39 increment, notarget replay
    new synthetic: pixmo exact-count variants, GUISyn negatives
Section 2 - GRPO prompts (what the policy rolls out on), one block per source_dataset
    from data_trimix_molmo_v2 (official 43-template pool), GT points overlaid in red.

    python3 scripts/render_v41_all_sources.py
"""
import base64
import collections
import html as _html
import io
import json
import random

import pandas as pd
from PIL import Image, ImageDraw

G = '/weka/oe-training-default/zixianm/yinuoy/grounding_rm'
OUT = '/weka/oe-training-default/zixianm/yinuoy/grounding-viz/v41_data.html'
rng = random.Random(11)


def b64(im, max_dim=760, q=68):
    im = im.copy()
    im.thumbnail((max_dim, max_dim))
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=q)
    return base64.b64encode(buf.getvalue()).decode()


def pair_card(r, tag):
    ci = 0 if r['label'] == 0 else 1
    m = r['meta']
    try:
        imc = Image.open(r['images'][ci]).convert('RGB')
        imr = Image.open(r['images'][1 - ci]).convert('RGB')
    except Exception:
        return None
    title = f"<b>{_html.escape(tag)}</b> | \u201c{_html.escape(str(m.get('phrase'))[:100])}\u201d"
    if m.get('k') is not None:
        title += f" | k={m.get('k')}\u2192{m.get('k_rej')}"
    prompt = r['prompt'] if isinstance(r['prompt'], str) else ''
    resp_c, resp_r = r['responses'][ci], r['responses'][1 - ci]
    def _txt(t, color):
        return (f'<div style="font-size:11.5px;color:{color};background:#f7f7f7;'
                f'border-radius:4px;padding:4px 6px;margin-top:3px">'
                f'<b>response:</b> {_html.escape(t[:400])}</div>')
    return ('<div style="border:1px solid #ccc;border-radius:8px;padding:8px;margin:10px 0">'
            f'<div style="font-size:13px;margin-bottom:4px">{title}</div>'
            f'<div style="font-size:11.5px;background:#eef4ff;border-radius:4px;padding:4px 6px;'
            f'margin-bottom:6px"><b>prompt:</b> {_html.escape(prompt[:300])}</div>'
            '<div style="display:flex;gap:8px">'
            f'<div style="flex:1"><img style="width:100%" src="data:image/jpeg;base64,{b64(imc)}">'
            '<div style="font-size:12px;color:#080"><b>CHOSEN</b></div>'
            f'{_txt(resp_c, "#060")}</div>'
            f'<div style="flex:1"><img style="width:100%" src="data:image/jpeg;base64,{b64(imr)}">'
            '<div style="font-size:12px;color:#a00"><b>REJECTED</b></div>'
            f'{_txt(resp_r, "#600")}</div></div></div>')


def rm_sections():
    out = ['<h2>1 \u00b7 v42 RM training pairs (som_marks_v2 renders + number-grounded text)</h2>']
    rows = [json.loads(l) for l in open(f'{G}/data/som_v42_numbered_training/train.jsonl')]
    bypool = collections.defaultdict(list)
    for r in rows:
        m = r['meta']
        if 'pool' in m:
            bypool[f"synth:{m['pool']}"].append(r)
        else:
            p = r['images'][0]
            for k in ('som_v2', 'som_v39', 'notarget'):
                if f'/{k}/' in p:
                    bypool[f'old_{k}_rerendered'].append(r)
                    break
    for pool in sorted(bypool):
        out.append(f'<h3>{_html.escape(pool)} ({len(bypool[pool]):,} rows)</h3>')
        n = 0
        for r in rng.sample(bypool[pool], min(30, len(bypool[pool]))):
            c = pair_card(r, pool)
            if c:
                out.append(c)
                n += 1
            if n >= 2:
                break
    return out


def grpo_section():
    out = ['<h2>2 · GRPO prompts (data_trimix_molmo_v2, official 43-template pool)</h2>'
           '<p>Red circle = reward_model.ground_truth (consumed only by GT-reward runs; the SoM arm never sees it). '
           'Rows with empty GT are tagged [no-target].</p>']
    df = pd.read_parquet(f'{G}/rl/data_trimix_molmo_v2/grounding_rl_train.parquet')
    # per-source composition table
    tot = len(df)
    rowshtml = []
    for src in sorted(df['source_dataset'].unique()):
        sub = df[df['source_dataset'] == src]
        nt = (sub['reward_model'].apply(lambda m: m['ground_truth'] == '[]')).mean()
        rowshtml.append(f'<tr><td>{_html.escape(src)}</td><td style="text-align:right">{len(sub):,}</td>'
                        f'<td style="text-align:right">{len(sub)/tot*100:.1f}%</td>'
                        f'<td style="text-align:right">{nt*100:.0f}%</td></tr>')
    out.append('<table style="border-collapse:collapse;margin:8px 0" border="1" cellpadding="5">'
               '<tr><th>source</th><th>rows</th><th>share</th><th>no-target rows</th></tr>'
               + ''.join(rowshtml) +
               f'<tr><th>TOTAL</th><th style="text-align:right">{tot:,}</th><th>100%</th><th></th></tr></table>')
    for src in sorted(df['source_dataset'].unique()):
        sub = df[df['source_dataset'] == src]
        out.append(f'<h3>{_html.escape(src)} ({len(sub):,} rows)</h3>')
        made = 0
        for _, r in sub.sample(min(60, len(sub)), random_state=11).iterrows():
            try:
                im = Image.open(r['image']).convert('RGB')
            except Exception:
                continue
            W, H = im.size
            gt = json.loads(r['reward_model']['ground_truth'])
            d = ImageDraw.Draw(im)
            for gx, gy in gt:
                x, y = gx / 1000 * W, gy / 1000 * H
                rr = max(8, int(0.012 * max(W, H)))
                d.ellipse([x - rr, y - rr, x + rr, y + rr], outline=(255, 0, 0),
                          width=max(3, rr // 3))
            tagg = '' if gt else ' <span style="color:#a60">[no-target]</span>'
            prompt = r['prompt'][0]['content'].replace('<image>', '')
            out.append('<div style="border:1px solid #ccc;border-radius:8px;padding:8px;margin:10px 0">'
                       f'<div style="font-size:13px;margin-bottom:4px">“{_html.escape(prompt[:160])}”{tagg}</div>'
                       f'<img style="max-width:70%" src="data:image/jpeg;base64,{b64(im)}"></div>')
            made += 1
            if made >= 6:
                break
    return out


def main():
    body = rm_sections() + grpo_section()
    page = ('<!DOCTYPE html><html><head><meta charset="utf-8"><title>v41 data — all sources</title></head>'
            '<body style="font-family:sans-serif;max-width:1150px;margin:0 auto;padding:20px">'
            '<h1>v41 data — all sources</h1>'
            '<p>RM pairs (chosen vs rejected) and GRPO prompts per source; every marker rendered with som_marks_v2.</p>'
            + '\n'.join(body) + '</body></html>')
    with open(OUT, 'w') as f:
        f.write(page)
    print(f'{OUT}: {len(page)/1e6:.1f} MB')


if __name__ == '__main__':
    main()
