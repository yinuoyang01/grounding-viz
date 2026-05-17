"""Generate cambrian_recap.html — inspection viz for 3-tier recaptioned Cambrian OCR data.

Source: /weka/oe-training-default/oe-encoder/cambrian_ocr/recap/cambrian_<ds>/<sample_id>/
  - <sample_id>.png (or .jpg)
  - caption.json  { high_level, mid_level, low_level }
  - _raw_input_facts.txt  (original QA facts fed to Qwen3-VL-8B)
"""

import os, json, random, base64, io
from html import escape
from PIL import Image

BASE = '/weka/oe-training-default/oe-encoder/cambrian_ocr/recap'
OUT = '/weka/oe-training-default/zixianm/yinuoy/grounding-viz/extras/dataset-inspect/cambrian_recap.html'

DATASETS = [
    ('cambrian_arxivqa',  'ArXiv QA — scientific paper figures (LaTeX, plots, equations)'),
    ('cambrian_ocr_vqa',  'OCR-VQA — book/document cover OCR + visual QA'),
    ('cambrian_screen_qa','ScreenQA — mobile/desktop UI screenshots'),
    ('cambrian_llavar',   'LLaVAR — text-rich images (signs, posters, packaging)'),
    ('cambrian_oodvqa',   'OOD-VQA — out-of-distribution multimodal QA'),
]

SAMPLES_PER_DS = 8
MAX_DIM = 480
random.seed(42)


def encode_image(path):
    try:
        with Image.open(path) as im:
            im = im.convert('RGB')
            w, h = im.size
            scale = MAX_DIM / max(w, h)
            if scale < 1:
                w, h = int(w * scale), int(h * scale)
                im = im.resize((w, h), Image.LANCZOS)
            buf = io.BytesIO()
            im.save(buf, format='JPEG', quality=85)
            return base64.b64encode(buf.getvalue()).decode(), w, h
    except Exception as e:
        return None, 0, 0


def sample_dataset(ds_dir, n):
    if not os.path.isdir(ds_dir):
        return []
    all_samples = os.listdir(ds_dir)
    random.shuffle(all_samples)
    out = []
    for sid in all_samples:
        if len(out) >= n: break
        d = os.path.join(ds_dir, sid)
        cap_path = os.path.join(d, 'caption.json')
        if not os.path.isfile(cap_path): continue
        # find image
        img_path = None
        for fn in os.listdir(d):
            if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_path = os.path.join(d, fn); break
        if not img_path: continue
        try:
            with open(cap_path) as fh: caps = json.load(fh)
            facts = ''
            facts_path = os.path.join(d, '_raw_input_facts.txt')
            if os.path.isfile(facts_path):
                with open(facts_path) as fh: facts = fh.read()
            out.append({'sid': sid, 'img_path': img_path, 'caps': caps, 'facts': facts})
        except Exception:
            continue
    return out


def render_sample(s):
    b64, w, h = encode_image(s['img_path'])
    if not b64: return ''
    caps = s['caps']
    parts = []
    for key, label in [('high_level','High-level'), ('mid_level','Mid-level'), ('low_level','Low-level')]:
        text = caps.get(key, '')
        if not text: continue
        text_esc = escape(str(text))
        n = len(text)
        if key == 'low_level' and n > 600:
            parts.append(f'<div class="section-title">{label} ({n} chars)</div>'
                         f'<details><summary class="caption-text" style="cursor:pointer">[click to expand]</summary>'
                         f'<div class="caption-text">{text_esc}</div></details>')
        else:
            parts.append(f'<div class="section-title">{label} ({n} chars)</div>'
                         f'<div class="caption-text">{text_esc}</div>')
    if s['facts'].strip():
        facts_esc = escape(s['facts'][:2000])
        parts.append(f'<div class="section-title">Raw QA facts (input)</div>'
                     f'<details><summary class="caption-text" style="cursor:pointer;opacity:.7">[click to expand]</summary>'
                     f'<pre class="facts">{facts_esc}</pre></details>')
    return f'''<div class="sample">
  <div><img src="data:image/jpeg;base64,{b64}" width="{w}" height="{h}"></div>
  <div>
    <div class="sample-meta">{escape(s['sid'])}</div>
    {"".join(parts)}
  </div>
</div>'''


def main():
    parts = ['''<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Cambrian OCR Recap Examples</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:0 auto;padding:24px;background:#FAF2E9;color:#0A3235;max-width:1400px;}
h1{margin:0 0 4px;}
h2{margin:32px 0 8px;border-bottom:2px solid #F0529C;padding-bottom:4px;}
.subtitle{color:rgba(10,50,53,0.55);margin-bottom:8px;font-size:14px;}
.ds-desc{background:#F1E4D1;padding:12px 16px;border-radius:8px;border-left:4px solid #105257;margin:8px 0 16px;font-size:13px;}
.sample{background:#FAF2E9;border:1px solid rgba(10,50,53,0.15);border-radius:12px;padding:16px;margin-bottom:16px;display:grid;grid-template-columns:520px 1fr;gap:16px;}
.sample img{max-width:100%;border-radius:8px;}
.sample-meta{font-size:11px;color:rgba(10,50,53,0.55);margin-bottom:6px;font-family:ui-monospace,Menlo,monospace;}
.section-title{font-size:11px;color:#F0529C;text-transform:uppercase;margin:12px 0 4px;font-weight:700;letter-spacing:.5px;}
.caption-text{font-size:14px;line-height:1.55;word-break:break-word;}
.facts{font-family:ui-monospace,Menlo,monospace;font-size:12px;line-height:1.5;background:#F1E4D1;padding:8px 12px;border-radius:6px;white-space:pre-wrap;}
</style></head><body>
<h1>Cambrian OCR — 3-tier Recaption Examples</h1>
<div class="subtitle">Recaptioned by Qwen3-VL-8B-Instruct, conditioned on image + original QA facts. ~150K samples across 5 datasets. Source: <code>/weka/oe-training-default/oe-encoder/cambrian_ocr/recap/</code></div>
<div class="subtitle">8 random samples per dataset · seed=42</div>
''']
    for ds, desc in DATASETS:
        ds_dir = os.path.join(BASE, ds)
        n_total = len(os.listdir(ds_dir)) if os.path.isdir(ds_dir) else 0
        print(f'[{ds}] sampling from {n_total} ...', flush=True)
        samples = sample_dataset(ds_dir, SAMPLES_PER_DS)
        parts.append(f'<h2>{ds} ({n_total:,} samples)</h2>')
        parts.append(f'<div class="ds-desc">{escape(desc)}</div>')
        for s in samples:
            parts.append(render_sample(s))
    parts.append('</body></html>')
    with open(OUT, 'w') as f: f.write('\n'.join(parts))
    sz = os.path.getsize(OUT) / 1024 / 1024
    print(f'\nWrote {OUT} ({sz:.1f} MB)')


if __name__ == '__main__':
    main()
