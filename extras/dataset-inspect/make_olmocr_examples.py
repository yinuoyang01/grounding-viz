"""Standalone olmOCR-mix-0225 viz — fast (no full tar index).

Strategy: open a couple of random tars, pick K PDFs from each, look up natural_text
from the matching parquet by (id, page_number). Avoids the slow 52-tar index build.
"""
import os, json, random, tarfile, base64, io
from glob import glob
from html import escape
import fitz
import pyarrow.parquet as pq

BASE = '/weka/oe-training-default/oe-encoder/olmocr_mix_0225'
OUT = '/weka/oe-training-default/zixianm/yinuoy/grounding-viz/extras/dataset-inspect/olmocr_examples.html'

# Sample from web (S2) PDF corpus only — that's the bulk of the dataset (240K of 256K pages)
SPLITS = [
    ('train-s2pdf.parquet',   'Web PDFs (academic 60%, brochure 12%, legal 11%, table 6%, diagram 5%, slideshow 2%)'),
    ('train-iabooks.parquet', 'Internet Archive books (public domain)'),
]

PER_SPLIT = 12
random.seed(7)


def load_parquet_lookup(path):
    """Build (id, page) -> response dict + url."""
    print(f'[load] {path}', flush=True)
    t = pq.read_table(path)
    ids = t['id'].to_pylist()
    pages = t['page_number'].to_pylist()
    urls = t['url'].to_pylist()
    resps = t['response'].to_pylist()
    lut = {}
    for pid, pn, url, r in zip(ids, pages, urls, resps):
        try:
            resp = json.loads(r) if isinstance(r, str) else r
        except Exception:
            resp = {}
        # parquet id is "<hash>-<page>"; PDF filename stem matches it exactly
        lut[pid] = {'page': pn, 'url': url, 'resp': resp}
    print(f'  {len(lut):,} rows loaded', flush=True)
    return lut


def sample_from_tars(lut, n, tar_glob, parse_key):
    """Pick n random tars, extract PDFs, render first page, look up text."""
    tars = sorted(glob(tar_glob))
    random.shuffle(tars)
    samples = []
    for tar_path in tars:
        if len(samples) >= n: break
        print(f'  reading {os.path.basename(tar_path)} ...', flush=True)
        try:
            with tarfile.open(tar_path) as tf:
                members = [m for m in tf if m.isfile() and m.name.endswith('.pdf')]
                random.shuffle(members)
                for m in members[:max(1, n // 2)]:  # take up to half from each tar
                    if len(samples) >= n: break
                    try:
                        key = os.path.splitext(os.path.basename(m.name))[0]  # "<hash>-<page>"
                        meta = lut.get(key)
                        if meta is None: continue
                        pdf_bytes = tf.extractfile(m).read()
                        doc = fitz.open(stream=pdf_bytes, filetype='pdf')
                        page = doc[0]
                        pix = page.get_pixmap(matrix=fitz.Matrix(1.4, 1.4))
                        img_jpeg = pix.tobytes('jpeg')
                        w, h = pix.width, pix.height
                        doc.close()
                        resp = meta['resp']
                        samples.append({
                            'name': m.name,
                            'id': key, 'page': meta['page'], 'url': meta['url'],
                            'img_jpeg': img_jpeg, 'w': w, 'h': h,
                            'text': resp.get('natural_text', '') or '',
                            'lang': resp.get('primary_language', ''),
                            'is_table': resp.get('is_table', False),
                            'is_diagram': resp.get('is_diagram', False),
                            'rotation': resp.get('rotation_correction', 0),
                        })
                    except Exception as e:
                        continue
        except Exception:
            continue
    return samples


def parse_pdf_key(name):
    """'abc123def-7.pdf' -> ('abc123def', 7)"""
    stem = os.path.splitext(os.path.basename(name))[0]
    pid, _, pn = stem.rpartition('-')
    return (pid, int(pn))


def render_sample(s):
    b64 = base64.b64encode(s['img_jpeg']).decode()
    text_esc = escape(s['text'])
    n = len(s['text'])
    flags = []
    if s['lang']: flags.append(f'lang={escape(str(s["lang"]))}')
    if s['is_table']: flags.append('<span style="color:#e11d48;font-weight:700">is_table</span>')
    if s['is_diagram']: flags.append('<span style="color:#e11d48;font-weight:700">is_diagram</span>')
    if s['rotation']: flags.append(f'rotation={s["rotation"]}°')
    flags_html = ' · '.join(flags) if flags else ''
    url_esc = escape(str(s['url']))
    if n > 1500:
        text_block = (f'<div class="section-title">natural_text ({n} chars)</div>'
                      f'<details><summary class="caption-text" style="cursor:pointer">[click to expand]</summary>'
                      f'<div class="caption-text" style="white-space:pre-wrap">{text_esc}</div></details>')
    else:
        text_block = (f'<div class="section-title">natural_text ({n} chars)</div>'
                      f'<div class="caption-text" style="white-space:pre-wrap">{text_esc}</div>')
    return f'''<div class="sample">
  <div><img src="data:image/jpeg;base64,{b64}" width="{s["w"]}" height="{s["h"]}" style="max-width:100%;border-radius:8px;border:1px solid rgba(10,50,53,0.15)"></div>
  <div>
    <div class="sample-meta">id={escape(s["id"])} · page={s["page"]} · {flags_html}</div>
    <div class="section-title">source url</div><div class="caption-text"><a href="{url_esc}" target="_blank" style="color:#105257;font-size:11px;word-break:break-all">{url_esc}</a></div>
    {text_block}
  </div>
</div>'''


def main():
    parts = ['''<!DOCTYPE html><html><head><meta charset="UTF-8"><title>olmOCR-mix-0225 Examples</title>
<style>
body{font-family:-apple-system,BlinkMacSystemFont,sans-serif;margin:0 auto;padding:24px;background:#FAF2E9;color:#0A3235;max-width:1400px;}
h1{margin:0 0 4px;}
h2{margin:32px 0 8px;border-bottom:2px solid #F0529C;padding-bottom:4px;}
.subtitle{color:rgba(10,50,53,0.55);margin-bottom:8px;font-size:14px;}
.ds-desc{background:#F1E4D1;padding:12px 16px;border-radius:8px;border-left:4px solid #105257;margin:8px 0 16px;font-size:13px;}
.sample{background:#FAF2E9;border:1px solid rgba(10,50,53,0.15);border-radius:12px;padding:16px;margin-bottom:16px;display:grid;grid-template-columns:520px 1fr;gap:16px;}
.sample-meta{font-size:11px;color:rgba(10,50,53,0.55);margin-bottom:6px;font-family:ui-monospace,Menlo,monospace;}
.section-title{font-size:11px;color:#F0529C;text-transform:uppercase;margin:12px 0 4px;font-weight:700;letter-spacing:.5px;}
.caption-text{font-size:13px;line-height:1.55;word-break:break-word;}
</style></head><body>
<h1>olmOCR-mix-0225 — Examples</h1>
<div class="subtitle">~250K PDF pages OCRed by GPT-4o-2024-08-06 (allenai/olmOCR-mix-0225). Source: <code>/weka/oe-training-default/oe-encoder/olmocr_mix_0225/</code></div>
''']
    for pq_name, desc in SPLITS:
        pq_path = os.path.join(BASE, pq_name)
        if not os.path.exists(pq_path):
            print(f'[skip] missing {pq_path}'); continue
        lut = load_parquet_lookup(pq_path)
        if pq_name == 'train-s2pdf.parquet':
            tar_glob = os.path.join(BASE, 'pdf_tarballs', 'pdf_chunk_*.tar.gz')
        else:
            # IA books are also in pdf_tarballs? Let's check — they may be the only tars containing those ids.
            tar_glob = os.path.join(BASE, 'pdf_tarballs', 'pdf_chunk_*.tar.gz')
        samples = sample_from_tars(lut, PER_SPLIT, tar_glob, parse_pdf_key)
        parts.append(f'<h2>{escape(pq_name)} ({len(lut):,} pages)</h2>')
        parts.append(f'<div class="ds-desc">{escape(desc)}</div>')
        for s in samples:
            parts.append(render_sample(s))
    parts.append('</body></html>')
    with open(OUT, 'w') as f: f.write('\n'.join(parts))
    sz = os.path.getsize(OUT) / 1024 / 1024
    print(f'\nWrote {OUT} ({sz:.1f} MB)')


if __name__ == '__main__':
    main()
