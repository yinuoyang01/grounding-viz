"""Static HTML visualizer: Grounding / OCR / Knowledge."""
import os, random, re, tarfile, json, io, base64, bisect, sys
from glob import glob
from html import escape
from PIL import Image

OUT = '/weka/oe-training-default/zixianm/yinuoy/grounding-viz/index.html'
SAMPLES = 15
MAX_DIM = 400

COLORS = ['#e53e3e','#38a169','#3182ce','#d69e2e','#805ad5','#dd6b20','#319795','#d53f8c',
          '#2b6cb0','#9f7aea','#e11d48','#16a34a','#2563eb','#ca8a04','#7c3aed','#ea580c',
          '#0d9488','#db2777','#1d4ed8','#8b5cf6']

BBOX_HEAD_RE = re.compile(r'^(.*?)\s*\(box\)\s*;\s*(.*)$')
POINT_HEAD_RE = re.compile(r'^(.*?)\s*\(point\)\s*;\s*(.*)$')
BBOX_ITEM_RE = re.compile(r'(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)')
POINT_ITEM_RE = re.compile(r'(\d+)\s*,\s*(\d+)')

# (path, description, sample_count, size_on_disk)
GROUNDING_TAR_DATASETS = {
    'GRIT': ('/weka/oe-training-default/oe-encoder/grit',
             'Large-scale bbox grounding from Kosmos-2 — phrase → multiple bboxes',
             '12.4M', '506 GB'),
    'Visual Genome': ('/weka/oe-training-default/oe-encoder/vg',
             'Dense region+object+attribute+relationship annotations per image',
             '110k', '16 GB'),
    'Open Images V7': ('/weka/oe-training-default/oe-encoder/openimages',
             'Google Open Images — 600+ categories, bbox + point + relations',
             '904k', '240 GB'),
    'Pixmo Point': ('/weka/oe-training-default/oe-encoder/pixmo',
             'Allen AI pointing data (used to train Molmo pointing)',
             '1.77M', '700 GB'),
}

WEB_GROUNDING_DATASETS = {
    'cc3m': ('/weka/oe-training-default/oe-encoder/cc3m/data',
             'Conceptual Captions 3M — web images + caption + LLM dense_caption + region points',
             '3.0M', '259 GB'),
    'cc12m': ('/weka/oe-training-default/oe-encoder/cc12m/data',
             'Conceptual Captions 12M — larger web image-caption corpus with regions',
             '11M', '1.1 TB'),
    'wit': ('/weka/oe-training-default/oe-encoder/wit/data',
             'Wikipedia Image-Text — long-tail knowledge, caption only (no regions)',
             '4.7M', '12 TB'),
    'SA-1B': ('/weka/oe-training-default/mm-olmo/img_datasets/SA-1B/data',
             'Meta Segment Anything 1B — diverse high-quality images + regions',
             '11.2M', '11 TB'),
    'yfcc15m': ('/weka/oe-training-default/mm-olmo/img_datasets/yfcc15m/data',
             'Flickr YFCC100M 15M subset — user-written title/description + regions',
             '14.8M', '1.8 TB'),
}

OCR_BASE = '/weka/oe-training-default/mm-olmo/molmo3_datasets/text_rich_caption'
# (description, sample_count, size_on_disk)
# sample_count = actual samples packed into tar (used in training), source dir count in parens
OCR_SUBSETS = {
    'chart': ('matplotlib/seaborn charts (bar/line/scatter/gantt) with 3-level captions', '355,488 (source 356,203)', '57 GB source / 26 GB tars'),
    'diagram': ('LaTeX/Mermaid diagrams (scientific/clinical/architecture)', '144,582', '23 GB source / 21 GB tars'),
    'doc': ('LaTeX / HTML rendered documents (forms/letters/reports)', '340,000 (source 440,337; pack interrupted at 80%)', '197 GB source / 148 GB tars'),
    'graphic': ('SVG graphics (icons, visual elements)', '89,260 (source 89,278)', '3.0 GB source / 2.2 GB tars'),
    'table': ('HTML-rendered tables with dense annotations', '317,563 (source 318,071)', '65 GB source / 33 GB tars'),
}

# Cambrian OCR datasets — recaptioned by Qwen3-VL-8B (3-tier high/mid/low), conditioned on image + original QA facts
CAMBRIAN_OCR_BASE = '/weka/oe-training-default/oe-encoder/cambrian_ocr/recap'
CAMBRIAN_OCR_SUBSETS = {
    'cambrian_arxivqa':   ('ArXiv QA — scientific paper figures (LaTeX, plots, equations); recaptioned w/ Qwen3-VL-8B + raw QA facts',  '44,026',  '35 GB'),
    'cambrian_ocr_vqa':   ('OCR-VQA — book/document cover OCR + visual QA; recaptioned w/ Qwen3-VL-8B + raw QA facts',                  '61,250',  '15 GB'),
    'cambrian_screen_qa': ('ScreenQA — mobile/desktop UI screenshots; recaptioned w/ Qwen3-VL-8B + raw QA facts',                       '25,391',  '14 GB'),
    'cambrian_llavar':    ('LLaVAR — text-rich images (signs, posters, packaging); recaptioned w/ Qwen3-VL-8B + raw QA facts',          '15,162',  '3.2 GB'),
    'cambrian_oodvqa':    ('OOD-VQA — out-of-distribution multimodal QA; recaptioned w/ Qwen3-VL-8B + raw QA facts',                    '5,400',   '1.6 GB'),
}

KNOWLEDGE_DATASETS = {
    'visual_knowledge_data_dump': ('/weka/oe-training-default/oe-encoder/visual_knowledge_data_dump',
            'Allen AI entity QA dump (S3 explore-multimodal-datasets/data-dumps/entities/) — named entity + question + answer + candidate image URLs. Packed ~2.3M (entity, image, answer) training samples (5 imgs/entity × ~460k entities w/ valid imgs; 0.2% PNGs corrupt → skipped)',
            '546,986 entities / ~2.3M training samples', '25 GB JSON + 1.6 TB images + 4.7 TB tars'),
}

# Grounding data filtering (Molmo2-4B + SAM2.1) — output JSONL preview
FILTER_DATASETS = {
    'Visual Genome': ('/weka/oe-training-default/oe-encoder/grounding_filter_molmo2/vg_filtered',
            '/weka/oe-training-default/oe-encoder/vg',
            'bbox GT — Hungarian F1 over per-pred ↔ box-center matching, in-box check'),
    'OpenImages': ('/weka/oe-training-default/oe-encoder/grounding_filter_molmo2/openimages_filtered',
            '/weka/oe-training-default/oe-encoder/openimages',
            'bbox GT — same as VG'),
    'GRIT': ('/weka/oe-training-default/oe-encoder/grounding_filter_molmo2/grit_filtered',
            '/weka/oe-training-default/oe-encoder/grit',
            'bbox GT — Kosmos-2 phrase grounding, noisier'),
    'Pixmo': ('/weka/oe-training-default/oe-encoder/grounding_filter_molmo2/pixmo_filtered',
            '/weka/oe-training-default/oe-encoder/pixmo',
            'point GT — per-instance SAM2.1 mask, pred ∈ mask check'),
}

# Legacy SOM RM preview (kept for back-compat but not displayed)
SOM_PREVIEW_DATASETS = {
    'VG smoke A (circle+number)': ('/weka/oe-training-default/zixianm/yinuoy/grounding_rm/data/pairs_smoke_som_preview/vg_k2_som_pairs.jsonl',
            'Style A: simple colored circle + number at each mark location. No SAM segmentation. Cheap rendering',
            '15 preview / 4k smoke', '~1.5 MB images (preview)'),
    'VG smoke B (SAM mask outline)': ('/weka/oe-training-default/zixianm/yinuoy/grounding_rm/data/pairs_smoke_som_b/vg_k2_som_pairs.jsonl',
            'Style B: SAM2.1 mask outline traced around object at each mark + numbered label at mask centroid. Stronger visual signal but ~10× slower (SAM forward per mark)',
            '2,000 (VG hard cases)', '~600 MB images'),
}


def encode_image(img_bytes, max_dim=MAX_DIM):
    try:
        img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    except Exception:
        return None, (0, 0)
    w, h = img.size
    if max(w, h) > max_dim:
        s = max_dim / max(w, h)
        img = img.resize((int(w*s), int(h*s)), Image.LANCZOS)
        w, h = img.size
    buf = io.BytesIO()
    img.save(buf, format='JPEG', quality=65, optimize=True)
    return base64.b64encode(buf.getvalue()).decode(), (w, h)


def parse_grounding_caption(c):
    c = c.strip()
    m = BBOX_HEAD_RE.match(c)
    if m:
        items = [{'y1': int(a), 'x1': int(b), 'y2': int(x), 'x2': int(y)} for a,b,x,y in BBOX_ITEM_RE.findall(m.group(2))]
        return {'phrase': m.group(1).strip(), 'kind': 'bbox', 'items': items}
    m = POINT_HEAD_RE.match(c)
    if m:
        items = [{'y': int(a), 'x': int(b)} for a,b in POINT_ITEM_RE.findall(m.group(2))]
        return {'phrase': m.group(1).strip(), 'kind': 'point', 'items': items}
    return None


def sample_tar_pairs(tar, n=1, skip=0):
    keys = {}
    seen = 0
    results = []
    for m in tar:
        if '.' not in m.name:
            continue
        k, ext = m.name.rsplit('.', 1)
        entry = keys.setdefault(k, {})
        entry[ext] = m
        if 'jpg' in entry and 'json' in entry:
            if seen < skip:
                seen += 1
                continue
            results.append((k, entry))
            if len(results) >= n:
                break
    return results


def count_tar_shards(root):
    return len(glob(os.path.join(root, '**/*.tar'), recursive=True))


def sample_grounding_tar(root, n):
    tars = sorted(glob(os.path.join(root, '**/*.tar'), recursive=True))
    if not tars:
        return [], 0
    random.shuffle(tars)
    samples = []
    seen = set()
    for tp in tars:
        if len(samples) >= n:
            break
        try:
            with tarfile.open(tp, 'r') as t:
                skip = random.randint(0, 200)
                for k, f in sample_tar_pairs(t, 1, skip=skip):
                    if (tp, k) in seen: continue
                    seen.add((tp, k))
                    img_bytes = t.extractfile(f['jpg']).read()
                    data = json.load(t.extractfile(f['json']))
                    caps = data.get('captions', [])
                    if not caps: continue
                    samples.append({'tar': os.path.basename(tp), 'key': k, 'img_bytes': img_bytes, 'captions': caps})
        except Exception:
            continue
    return samples[:n], len(tars)


def render_grounding_tar_sample(s, sid):
    img_b64, (w, h) = encode_image(s['img_bytes'])
    if not img_b64: return ''
    parsed = [(i, c, parse_grounding_caption(c)) for i, c in enumerate(s['captions'])]
    svg = []
    for i, c, p in parsed:
        if not p or not p.get('items'): continue
        color = COLORS[i % len(COLORS)]
        phrase_attr = escape(p['phrase'], quote=True)
        for j, it in enumerate(p['items']):
            bid = f's{sid}_{i}_{j}'
            if p['kind'] == 'bbox':
                x1 = it['x1']/999*w; y1 = it['y1']/999*h
                x2 = it['x2']/999*w; y2 = it['y2']/999*h
                svg.append(f'<rect id="{bid}" class="box s{sid}" data-idx="{i}" data-phrase="{phrase_attr}" x="{x1:.1f}" y="{y1:.1f}" width="{(x2-x1):.1f}" height="{(y2-y1):.1f}" stroke="{color}" fill="transparent" stroke-width="2"></rect>')
            else:
                px = it['x']/999*w; py = it['y']/999*h
                r = max(4, int(min(w, h)*0.018))
                svg.append(f'<circle id="{bid}" class="box s{sid}" data-idx="{i}" data-phrase="{phrase_attr}" cx="{px:.1f}" cy="{py:.1f}" r="{r}" stroke="yellow" fill="{color}" stroke-width="1"></circle>')
    lines = []
    for i, c, p in parsed:
        color = COLORS[i % len(COLORS)]
        safe = escape(c)
        safe = re.sub(r'\(box\)', r'<span class="tag-bbox">(box)</span>', safe)
        safe = re.sub(r'\(point\)', r'<span class="tag-pt">(point)</span>', safe)
        lines.append(f'<div class="cap" data-sample="{sid}" data-idx="{i}" style="border-left: 3px solid {color}">{safe}</div>')
    return f'''<div class="sample" data-sample-idx="{sid}">
  <div class="img-wrap" style="width:{w}px;height:{h}px">
    <img src="data:image/jpeg;base64,{img_b64}" width="{w}" height="{h}">
    <svg class="overlay" viewBox="0 0 {w} {h}">{''.join(svg)}</svg>
    <div class="tooltip"></div>
  </div>
  <div>
    <div class="sample-meta">{escape(s['tar'])}/{escape(s['key'])} · {len(s['captions'])} captions</div>
    <div class="caption-list">{''.join(lines)}</div>
  </div>
</div>'''


def sample_web_grounding(root, n):
    tars = sorted(glob(os.path.join(root, '*.tar')))
    if not tars:
        return [], 0
    random.shuffle(tars)
    samples = []
    seen = set()
    for tp in tars[:n*2]:
        if len(samples) >= n: break
        try:
            with tarfile.open(tp, 'r') as t:
                skip = random.randint(0, 100)
                for k, f in sample_tar_pairs(t, 1, skip=skip):
                    if (tp, k) in seen: continue
                    seen.add((tp, k))
                    img_bytes = t.extractfile(f['jpg']).read()
                    data = json.load(t.extractfile(f['json']))
                    if data.get('status') and data['status'] != 'success': continue
                    samples.append({'tar': os.path.basename(tp), 'key': k, 'img_bytes': img_bytes,
                                    'caption': data.get('caption') or data.get('global_caption', ''),
                                    'dense_caption': data.get('dense_caption', ''),
                                    'regions': data.get('regions', []),
                                    'url': data.get('url', '')})
        except Exception:
            continue
    return samples[:n], len(tars)


def render_web_grounding_sample(s, sid):
    img_b64, (w, h) = encode_image(s['img_bytes'])
    if not img_b64: return ''
    cap = escape(s.get('caption', '') or '(no caption)')
    dense = escape(s.get('dense_caption', '') or '')
    dense_block = f'<div class="section-title">dense_caption</div><div class="caption-text">{dense}</div>' if dense else ''
    regions = s.get('regions') or []
    svg_items, region_lines = [], []
    for i, r in enumerate(regions):
        if not isinstance(r, dict): continue
        rc = r.get('region_caption', '')
        pts = r.get('points') or []
        color = COLORS[i % len(COLORS)]
        phrase_attr = escape(rc, quote=True)
        for j, pt in enumerate(pts):
            try:
                px = float(pt['x'])/100*w; py = float(pt['y'])/100*h
            except Exception:
                continue
            rr = max(4, int(min(w, h)*0.018))
            svg_items.append(f'<circle id="s{sid}_{i}_{j}" class="box s{sid}" data-idx="{i}" data-phrase="{phrase_attr}" cx="{px:.1f}" cy="{py:.1f}" r="{rr}" stroke="yellow" fill="{color}" stroke-width="1"></circle>')
        region_lines.append(f'<div class="cap" data-sample="{sid}" data-idx="{i}" style="border-left: 3px solid {color}">{escape(rc)} <span class="tag-pt">({len(pts)} pt)</span></div>')
    svg_block = tooltip_block = regions_block = ''
    if svg_items:
        svg_block = f'<svg class="overlay" viewBox="0 0 {w} {h}">{"".join(svg_items)}</svg>'
        tooltip_block = '<div class="tooltip"></div>'
        regions_block = f'<div class="section-title">regions ({len(regions)})</div><div class="caption-list">{"".join(region_lines)}</div>'
    return f'''<div class="sample" data-sample-idx="{sid}">
  <div class="img-wrap" style="width:{w}px;height:{h}px">
    <img src="data:image/jpeg;base64,{img_b64}" width="{w}" height="{h}">
    {svg_block}{tooltip_block}
  </div>
  <div>
    <div class="sample-meta">{escape(s['tar'])}/{escape(s['key'])}</div>
    <div class="section-title">caption</div><div class="caption-text">{cap}</div>
    {dense_block}{regions_block}
  </div>
</div>'''


def sample_ocr_subset(subset_dir, n, max_scan=80):
    all_files = []
    try:
        top = list(os.scandir(subset_dir))
    except Exception:
        return [], 0
    total_dirs = len(top)
    random.shuffle(top)
    for e in top[:3]:
        if len(all_files) >= max_scan: break
        try:
            if e.is_file() and e.name.lower().endswith(('.jpg','.jpeg','.png')):
                all_files.append(e.path)
            elif e.is_dir():
                count = 0
                with os.scandir(e.path) as sub:
                    for se in sub:
                        if count >= max_scan: break
                        if se.is_file() and se.name.lower().endswith(('.jpg','.jpeg','.png')):
                            all_files.append(se.path)
                            count += 1
        except Exception:
            continue
    if not all_files: return [], total_dirs
    random.shuffle(all_files)
    samples = []
    for img_path in all_files:
        if len(samples) >= n: break
        try:
            with open(img_path, 'rb') as fh: img_bytes = fh.read()
            img_dir = os.path.dirname(img_path)
            annotations = []
            for fname in ('caption.json', 'metadata.json'):
                p = os.path.join(img_dir, fname)
                if os.path.isfile(p):
                    try:
                        with open(p, 'r', errors='replace') as fh: content = fh.read()
                        annotations.append((fname, content))
                    except Exception: pass
            samples.append({'path': os.path.relpath(img_path, subset_dir), 'img_bytes': img_bytes, 'annotations': annotations})
        except Exception:
            continue
    return samples, total_dirs


# olmOCR-mix-0225 — parquet rows + PDFs in tarballs (52 chunks, alphabetically sharded)
OLMOCR_BASE = '/weka/oe-training-default/oe-encoder/olmocr_mix_0225'
OLMOCR_TAR_INDEX = None  # lazy-built: list of (first_key, tar_path)

def _build_olmocr_tar_index():
    global OLMOCR_TAR_INDEX
    if OLMOCR_TAR_INDEX is not None: return OLMOCR_TAR_INDEX
    print('  [olmocr] building tar index (one-time)...')
    tars = sorted(glob(os.path.join(OLMOCR_BASE, 'pdf_tarballs', 'pdf_chunk_*.tar.gz')))
    idx = []
    for t in tars:
        try:
            with tarfile.open(t) as tf:
                m = next((m for m in tf if m.isfile() and m.name.endswith('.pdf')), None)
                if m: idx.append((m.name, t))
        except Exception: continue
    idx.sort()
    OLMOCR_TAR_INDEX = idx
    return idx

def _extract_pdf_from_tars(key):
    """Find PDF by key in alphabetically sharded tars (binary search on tar's first key)."""
    idx = _build_olmocr_tar_index()
    keys = [k for k,_ in idx]
    i = bisect.bisect_right(keys, key) - 1
    if i < 0: return None
    # Try this tar and a couple nearby (in case of edge cases)
    for j in (i, i+1):
        if j < 0 or j >= len(idx): continue
        try:
            with tarfile.open(idx[j][1]) as tf:
                m = tf.getmember(key)
                return tf.extractfile(m).read()
        except KeyError: continue
        except Exception: continue
    return None

def sample_olmocr_subset(parquet_name, n):
    """Fast: random tars → extract PDFs → lookup by parquet id (which is '<hash>-<page>')."""
    import pyarrow.parquet as pq
    try:
        import fitz
    except ImportError:
        print('  [olmocr] WARN: pymupdf not installed, skipping')
        return [], 0
    pq_path = os.path.join(OLMOCR_BASE, parquet_name)
    if not os.path.exists(pq_path): return [], 0
    t = pq.read_table(pq_path)
    total = t.num_rows
    # Build id -> {page, url, resp} lookup
    lut = {}
    for pid, pn, url, r in zip(t['id'].to_pylist(), t['page_number'].to_pylist(),
                                t['url'].to_pylist(), t['response'].to_pylist()):
        try:
            resp = json.loads(r) if isinstance(r, str) else r
        except Exception:
            resp = {}
        lut[pid] = {'page': pn, 'url': url, 'resp': resp}
    tars = sorted(glob(os.path.join(OLMOCR_BASE, 'pdf_tarballs', 'pdf_chunk_*.tar.gz')))
    rng = random.Random(7)
    rng.shuffle(tars)
    samples = []
    for tar_path in tars:
        if len(samples) >= n: break
        try:
            with tarfile.open(tar_path) as tf:
                members = [m for m in tf if m.isfile() and m.name.endswith('.pdf')]
                rng.shuffle(members)
                for m in members[:max(1, n // 2)]:
                    if len(samples) >= n: break
                    try:
                        key = os.path.splitext(os.path.basename(m.name))[0]
                        meta = lut.get(key)
                        if meta is None: continue
                        pdf_bytes = tf.extractfile(m).read()
                        doc = fitz.open(stream=pdf_bytes, filetype='pdf')
                        page = doc[0]
                        pix = page.get_pixmap(matrix=fitz.Matrix(1.0, 1.0))
                        img_jpeg = pix.tobytes('jpeg')
                        w, h = pix.width, pix.height
                        doc.close()
                        resp = meta['resp']
                        samples.append({
                            'id': key, 'page': meta['page'], 'url': meta['url'],
                            'img_jpeg': img_jpeg, 'w': w, 'h': h,
                            'text': resp.get('natural_text', '') or '',
                            'lang': resp.get('primary_language', ''),
                            'is_table': resp.get('is_table', False),
                            'is_diagram': resp.get('is_diagram', False),
                            'rotation': resp.get('rotation_correction', 0),
                        })
                    except Exception: continue
        except Exception: continue
    return samples, total

def render_olmocr_sample(s, sid):
    img_b64 = base64.b64encode(s['img_jpeg']).decode()
    text_esc = escape(s['text'])
    n = len(s['text'])
    flags = []
    if s['lang']: flags.append(f'lang={escape(str(s["lang"]))}')
    if s['is_table']: flags.append('<span style="color:#e11d48;font-weight:700">is_table</span>')
    if s['is_diagram']: flags.append('<span style="color:#e11d48;font-weight:700">is_diagram</span>')
    if s['rotation']: flags.append(f'rotation={s["rotation"]}°')
    flags_html = ' · '.join(flags)
    url_esc = escape(str(s['url']))
    if len(s['text']) > 800:
        text_block = (f'<div class="section-title">natural_text ({n} chars)</div>'
                      f'<details><summary class="caption-text" style="cursor:pointer">[expand]</summary>'
                      f'<div class="caption-text" style="white-space:pre-wrap">{text_esc}</div></details>')
    else:
        text_block = (f'<div class="section-title">natural_text ({n} chars)</div>'
                      f'<div class="caption-text" style="white-space:pre-wrap">{text_esc}</div>')
    return f'''<div class="sample" data-sample-idx="{sid}">
  <div class="img-wrap">
    <img src="data:image/jpeg;base64,{img_b64}" style="max-width:100%;height:auto;border-radius:8px;border:1px solid rgba(10,50,53,0.15)">
  </div>
  <div>
    <div class="sample-meta">id={escape(s["id"])} · page={s["page"]} · {flags_html}</div>
    <div class="section-title">source url</div><div class="caption-text"><a href="{url_esc}" target="_blank" style="color:#105257;font-size:11px;word-break:break-all">{url_esc}</a></div>
    {text_block}
  </div>
</div>'''


OLMOCR_SUBSETS = {
    'olmocr_s2pdf':   ('Web-crawled PDFs (S2) — OCRed by GPT-4o-2024-08-06; ~60% academic, 12% brochure, 11% legal',  '239,774 pages (99,903 docs)',  '~52 GB pdf_tarballs + 0.34 GB parquet',  'train-s2pdf.parquet'),
    'olmocr_iabooks': ('Internet Archive books (public domain) — OCRed by GPT-4o-2024-08-06',                          '16,803 pages (5,601 docs)',    '~0.02 GB parquet',                       'train-iabooks.parquet'),
}

# TextOCR / TextCaps — OpenImages subset with OCR word annotations + natural-language captions
TEXTOCR_BASE = '/weka/oe-training-default/oe-encoder/textocr'
TEXTCAPS_BASE = '/weka/oe-training-default/oe-encoder/textcaps'

def sample_textocr(n):
    """Sample n images from TextOCR train; each image has ~50 OCR word boxes."""
    import json as _json
    json_path = os.path.join(TEXTOCR_BASE, 'TextOCR_0.1_train.json')
    if not os.path.exists(json_path): return [], 0
    with open(json_path) as fh: d = _json.load(fh)
    imgs = list(d['imgs'].values())
    total = len(imgs)
    rng = random.Random(11)
    rng.shuffle(imgs)
    samples = []
    for img in imgs:
        if len(samples) >= n: break
        # file_name like "train/<id>.jpg" → actual at images/train_images/<id>.jpg
        fid = os.path.basename(img['file_name']).rsplit('.', 1)[0]
        path = os.path.join(TEXTOCR_BASE, 'images', 'train_images', f'{fid}.jpg')
        if not os.path.exists(path): continue
        ann_ids = d['imgToAnns'].get(img['id'], [])
        anns = [d['anns'][a] for a in ann_ids if a in d['anns']]
        try:
            with open(path, 'rb') as fh: img_bytes = fh.read()
        except Exception: continue
        samples.append({'img_bytes': img_bytes, 'img_w': img['width'], 'img_h': img['height'],
                        'id': fid, 'anns': anns})
    return samples, total

def render_textocr_sample(s, sid):
    b64, (w, h) = encode_image(s['img_bytes'])
    if not b64: return ''
    scale_x = w / max(1, s['img_w']); scale_y = h / max(1, s['img_h'])
    boxes = []
    for a in s['anns'][:200]:
        x, y, bw, bh = a['bbox']
        if a.get('utf8_string') == '.': continue  # skip illegible markers
        boxes.append(f'<rect x="{x*scale_x:.1f}" y="{y*scale_y:.1f}" width="{bw*scale_x:.1f}" height="{bh*scale_y:.1f}" fill="none" stroke="#e11d48" stroke-width="1.5"/>')
    svg = f'<svg class="overlay" viewBox="0 0 {w} {h}" style="position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none">{"".join(boxes)}</svg>'
    words = [escape(a.get('utf8_string', '')) for a in s['anns'][:50] if a.get('utf8_string') and a['utf8_string'] != '.']
    words_html = ' · '.join(words)
    return f'''<div class="sample" data-sample-idx="{sid}">
  <div class="img-wrap" style="position:relative">
    <img src="data:image/jpeg;base64,{b64}" style="max-width:100%;height:auto;border-radius:8px;border:1px solid rgba(10,50,53,0.15)">
    {svg}
  </div>
  <div>
    <div class="sample-meta">id={escape(s["id"])} · {len(s["anns"])} word boxes</div>
    <div class="section-title">OCR words (first 50)</div>
    <div class="caption-text" style="font-family:ui-monospace,Menlo,monospace;font-size:12px;line-height:1.7">{words_html}</div>
  </div>
</div>'''

def sample_textcaps(n):
    """Sample n images from TextCaps train; each has 5 reference captions."""
    import json as _json
    json_path = os.path.join(TEXTCAPS_BASE, 'TextCaps_0.1_train.json')
    if not os.path.exists(json_path): return [], 0
    with open(json_path) as fh: d = _json.load(fh)
    rows = d['data']
    total = len(rows)
    # one image may appear multiple times (one per caption); de-dup
    seen = {}
    for r in rows:
        if r['image_id'] not in seen: seen[r['image_id']] = r
    images = list(seen.values())
    rng = random.Random(13)
    rng.shuffle(images)
    samples = []
    for r in images:
        if len(samples) >= n: break
        # textcaps images symlink to textocr/images/train_images
        path = os.path.join(TEXTCAPS_BASE, 'images', 'train_images', f'{r["image_id"]}.jpg')
        if not os.path.exists(path):
            # fallback to textocr direct
            path = os.path.join(TEXTOCR_BASE, 'images', 'train_images', f'{r["image_id"]}.jpg')
        if not os.path.exists(path): continue
        try:
            with open(path, 'rb') as fh: img_bytes = fh.read()
        except Exception: continue
        samples.append({'img_bytes': img_bytes, 'id': r['image_id'],
                        'classes': r.get('image_classes', []),
                        'captions': r.get('reference_strs', [])})
    return samples, total

def render_textcaps_sample(s, sid):
    b64, (w, h) = encode_image(s['img_bytes'])
    if not b64: return ''
    caps_html = ''.join(f'<div class="caption-text" style="margin-bottom:6px;padding:6px 10px;background:#F1E4D1;border-radius:4px">{escape(c)}</div>' for c in s['captions'])
    classes_html = ', '.join(escape(c) for c in s.get('classes', []))
    return f'''<div class="sample" data-sample-idx="{sid}">
  <div class="img-wrap">
    <img src="data:image/jpeg;base64,{b64}" style="max-width:100%;height:auto;border-radius:8px;border:1px solid rgba(10,50,53,0.15)">
  </div>
  <div>
    <div class="sample-meta">id={escape(s["id"])} · classes: {classes_html}</div>
    <div class="section-title">5 reference captions</div>
    {caps_html}
  </div>
</div>'''

TEXTOCR_CAPS_SUBSETS = {
    'textocr':  ('TextOCR (Singh et al, CVPR\'21) — OCR word-level annotations on OpenImages subset; ~50 words/image',  '21,778 images / 1.05M word anns',  '7.0 GB images + 0.32 GB json'),
    'textcaps': ('TextCaps (Sidorov et al, ECCV\'20) — natural-language captions describing text-rich OpenImages; 5 captions/image',  '21,953 images / 5x captions',  '0.21 GB json (shares OpenImages images w/ textocr)'),
}


def sample_cambrian_subset(subset_dir, n):
    """Cambrian recap: each sample is one dir with caption.json + one image + raw txts."""
    if not os.path.isdir(subset_dir):
        return [], 0
    all_dirs = os.listdir(subset_dir)
    total = len(all_dirs)
    random.shuffle(all_dirs)
    samples = []
    for sid in all_dirs:
        if len(samples) >= n: break
        d = os.path.join(subset_dir, sid)
        cap_path = os.path.join(d, 'caption.json')
        if not os.path.isfile(cap_path): continue
        img_path = None
        try:
            for fn in os.listdir(d):
                if fn.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(d, fn); break
        except Exception:
            continue
        if not img_path: continue
        try:
            with open(img_path, 'rb') as fh: img_bytes = fh.read()
            annotations = [('caption.json', open(cap_path).read())]
            samples.append({'path': sid, 'img_bytes': img_bytes, 'annotations': annotations})
        except Exception:
            continue
    return samples, total


def render_ocr_sample(s, sid):
    img_b64, (w, h) = encode_image(s['img_bytes'])
    if not img_b64: return ''
    caption_obj, meta_obj = {}, {}
    for fname, content in s.get('annotations', []):
        try:
            obj = json.loads(content)
            if fname == 'caption.json': caption_obj = obj
            elif fname == 'metadata.json': meta_obj = obj
        except Exception: pass
    parts = []
    for key in ('high_level', 'mid_level', 'low_level'):
        if key in caption_obj and caption_obj[key]:
            text = escape(str(caption_obj[key]))
            if key == 'low_level' and len(text) > 400:
                parts.append(f'<div class="section-title">{key} ({len(text)} chars)</div>'
                             f'<details><summary class="caption-text">[expand]</summary><div class="caption-text">{text}</div></details>')
            else:
                parts.append(f'<div class="section-title">{key}</div><div class="caption-text">{text}</div>')
    if meta_obj:
        meta_small = {k: meta_obj.get(k) for k in ('persona','model_name','content_type','description') if meta_obj.get(k)}
        if meta_small:
            rows = ''.join(f'<div><b>{escape(k)}:</b> {escape(str(v)[:300])}</div>' for k,v in meta_small.items())
            parts.append(f'<div class="section-title">metadata</div><div class="caption-text">{rows}</div>')
    if not parts:
        parts.append('<div class="caption-text" style="opacity:.6">(no caption.json/metadata.json found)</div>')
    return f'''<div class="sample" data-sample-idx="{sid}">
  <div class="img-wrap" style="width:{w}px;height:{h}px">
    <img src="data:image/jpeg;base64,{img_b64}" width="{w}" height="{h}">
  </div>
  <div>
    <div class="sample-meta">{escape(s['path'])}</div>
    {''.join(parts)}
  </div>
</div>'''


def sample_knowledge(root, n):
    files = sorted(glob(os.path.join(root, '*.json')))
    if not files: return [], 0
    samples = []
    for f in files[:3]:
        try:
            with open(f) as fh: data = json.load(fh)
        except Exception: continue
        if isinstance(data, list):
            random.shuffle(data)
            for obj in data:
                if len(samples) >= n: break
                samples.append({'source': os.path.basename(f), 'entry': obj})
        if len(samples) >= n: break
    return samples[:n], len(files)


def render_knowledge_sample(s, sid):
    entry = s['entry']
    named = escape(str(entry.get('namedEntity', '')))
    q = escape(str(entry.get('question', '')))
    a = escape(str(entry.get('answer', ''))[:1500])
    sel = entry.get('selectedImages', {})
    urls = list(sel.keys()) if isinstance(sel, dict) else []
    # Image grid on left (max 12 thumbnails), text on right
    img_tiles = ''.join(f'<img class="knowledge-thumb" src="{escape(u)}" referrerpolicy="no-referrer" loading="lazy" onerror="this.style.display=\'none\'">' for u in urls[:12])
    return f'''<div class="sample knowledge-sample" data-sample-idx="{sid}">
  <div class="knowledge-img-grid">
    <div class="sample-meta">{len(urls)} images</div>
    <div class="thumb-grid">{img_tiles}</div>
  </div>
  <div>
    <div class="sample-meta">{escape(s['source'])} · {entry.get('id', '')}</div>
    <div class="section-title">namedEntity</div><div class="caption-text"><b>{named}</b></div>
    <div class="section-title">question</div><div class="caption-text">{q}</div>
    <div class="section-title">answer</div><div class="caption-text">{a}</div>
  </div>
</div>'''


def sample_som_jsonl(jsonl_path, n):
    """Read SOM K=2 jsonl, return up to n records (with image path)."""
    if not os.path.isfile(jsonl_path):
        return [], 0
    records = []
    with open(jsonl_path) as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except Exception:
                continue
    total = len(records)
    random.shuffle(records)
    samples = []
    for rec in records[:n]:
        img_path = rec.get('image', '')
        if not os.path.isfile(img_path):
            continue
        try:
            with open(img_path, 'rb') as fh:
                img_bytes = fh.read()
        except Exception:
            continue
        samples.append({'rec': rec, 'img_bytes': img_bytes})
    return samples, total


def sample_filter_jsonl(jsonl_dir, tar_root, n, want_error=None):
    """Read filter jsonl + fetch image bytes from tar.

    want_error: if True, only error=True samples; if False, only error=False; None = mix.
    """
    import glob as _glob
    files = sorted(_glob.glob(os.path.join(jsonl_dir, 'worker_*.jsonl')))
    if not files:
        return [], 0
    candidates = []
    for f in files:
        for line in open(f):
            try:
                r = json.loads(line)
            except Exception:
                continue
            if want_error is not None and bool(r.get('error')) != want_error:
                continue
            candidates.append(r)
            if len(candidates) >= n * 30:
                break
        if len(candidates) >= n * 30:
            break
    random.shuffle(candidates)

    # Build basename → list-of-paths index for nested tar layouts (e.g. GRIT coyo_*_snappy/)
    all_tars = sorted(glob(os.path.join(tar_root, '**/*.tar'), recursive=True))
    tar_index = {}
    for tp in all_tars:
        tar_index.setdefault(os.path.basename(tp), []).append(tp)

    # Group by shard for fast tar read
    by_shard = {}
    for rec in candidates:
        by_shard.setdefault(rec['shard'], []).append(rec)

    samples = []
    for shard_name, recs in by_shard.items():
        tar_paths = tar_index.get(shard_name, [])
        flat = os.path.join(tar_root, shard_name)
        if os.path.isfile(flat) and flat not in tar_paths:
            tar_paths = [flat] + tar_paths
        if not tar_paths:
            continue
        wanted_keys = {r['key']: r for r in recs}
        for tar_path in tar_paths:
            if not wanted_keys:
                break
            try:
                with tarfile.open(tar_path, 'r') as t:
                    for m in t:
                        k = m.name.rsplit('.', 1)[0]
                        if k in wanted_keys and m.name.endswith('.jpg'):
                            img_bytes = t.extractfile(m).read()
                            rec = wanted_keys.pop(k)
                            samples.append({'rec': rec, 'img_bytes': img_bytes})
                            if not wanted_keys:
                                break
                            if len(samples) >= n:
                                break
            except Exception:
                continue
            if len(samples) >= n:
                break
        if len(samples) >= n:
            break
    return samples[:n], len(candidates)


def render_filter_sample(s, sid):
    rec = s['rec']
    img_b64, (w, h) = encode_image(s['img_bytes'])
    if not img_b64: return ''
    info = rec.get('info', {})
    phrase = rec.get('phrase', '')
    gt_items = rec.get('gt', [])
    pred_pts = rec.get('pred', [])
    kind = rec.get('kind', '')
    err = rec.get('error', False)
    f1 = info.get('f1', 0)
    p = info.get('precision', 0)
    r_score = info.get('recall', 0)
    n_gt = info.get('n_gt', len(gt_items))
    n_pred = info.get('n_pred', len(pred_pts))
    matched = info.get('matched_correct', 0)

    # Source image dims (recover from pred + pred_raw, similar to filter logic)
    pred_raw = rec.get('pred_raw', '')
    pm = re.search(r'(\d+) (\d{3,4}) (\d{3,4})', pred_raw)
    if pm and pred_pts:
        raw_x, raw_y = float(pm.group(2)), float(pm.group(3))
        if raw_x > 0 and raw_y > 0:
            src_w = pred_pts[0][0] / (raw_x / 1000.0)
            src_h = pred_pts[0][1] / (raw_y / 1000.0)
            scale_x = w / src_w
            scale_y = h / src_h
        else:
            scale_x = scale_y = 1.0
            src_w = src_h = 0
    else:
        scale_x = scale_y = 1.0
        src_w = src_h = 0

    svg_items = []
    # Draw GT in green
    if kind == 'bbox':
        for i, (y1, x1, y2, x2) in enumerate(gt_items):
            bx1 = x1 / 999 * w
            by1 = y1 / 999 * h
            bx2 = x2 / 999 * w
            by2 = y2 / 999 * h
            svg_items.append(f'<rect x="{bx1:.1f}" y="{by1:.1f}" '
                             f'width="{(bx2-bx1):.1f}" height="{(by2-by1):.1f}" '
                             f'stroke="#16a34a" fill="transparent" stroke-width="2" '
                             f'data-label="GT box {i+1}"></rect>')
    else:  # point
        for i, (gy, gx) in enumerate(gt_items):
            px_x = gx / 999 * w
            px_y = gy / 999 * h
            svg_items.append(f'<circle cx="{px_x:.1f}" cy="{px_y:.1f}" r="5" '
                             f'fill="#16a34a" stroke="white" stroke-width="1" '
                             f'data-label="GT pt {i+1}"></circle>')
    # Draw pred in red (rescaled if image dims recovered)
    for i, (px, py) in enumerate(pred_pts):
        if src_w > 0:
            cx = px * scale_x
            cy = py * scale_y
        else:
            cx = px
            cy = py
        svg_items.append(f'<circle cx="{cx:.1f}" cy="{cy:.1f}" r="5" '
                         f'fill="#e53e3e" stroke="white" stroke-width="1" '
                         f'data-label="pred {i+1}"></circle>')

    err_badge = '❌ HARD' if err else '✅ OK'
    badge_color = 'var(--accent)' if err else 'var(--accent2)'
    return f'''<div class="sample" data-sample-idx="{sid}">
  <div class="img-wrap" style="width:{w}px;height:{h}px">
    <img src="data:image/jpeg;base64,{img_b64}" width="{w}" height="{h}">
    <svg class="overlay" viewBox="0 0 {w} {h}">{''.join(svg_items)}</svg>
  </div>
  <div>
    <div class="sample-meta">{escape(rec.get('shard',''))}/{escape(rec.get('key',''))} · kind={kind}</div>
    <div class="section-title">phrase</div>
    <div class="caption-text"><b>{escape(phrase)}</b></div>
    <div class="section-title">verdict</div>
    <div class="caption-text" style="color:{badge_color};font-weight:600">{err_badge}</div>
    <div class="section-title">metrics</div>
    <div class="caption-text">F1={f1:.2f} · P={p:.2f} · R={r_score:.2f}<br>matched={matched}/{n_gt} (n_pred={n_pred})</div>
    <div class="section-title">overlay</div>
    <div class="caption-text"><span style="color:#16a34a;font-weight:600">●</span> GT &nbsp;&nbsp; <span style="color:#e53e3e;font-weight:600">●</span> Molmo pred</div>
    <div class="section-title">pred_raw</div>
    <div class="caption-text" style="font-family:ui-monospace;font-size:11px">{escape(pred_raw[:160])}</div>
  </div>
</div>'''


def render_som_sample(s, sid):
    rec = s['rec']
    img_b64, (w, h) = encode_image(s['img_bytes'])
    if not img_b64: return ''
    label_idx = rec.get('label', 0)
    gt_mark = label_idx + 1
    prompt = escape(rec.get('prompt', ''))
    responses = rec.get('responses', [])
    meta = rec.get('meta', {})
    resp_html = ''.join(
        f'<div class="cap" style="border-left:3px solid {COLORS[i % len(COLORS)]}">{escape(r)} '
        f'{"<b>(GT)</b>" if i == label_idx else ""}</div>'
        for i, r in enumerate(responses))
    gt = meta.get('gt_norm', [0, 0])
    wrong = meta.get('wrong_norm', [0, 0])
    return f'''<div class="sample" data-sample-idx="{sid}">
  <div class="img-wrap" style="width:{w}px;height:{h}px">
    <img src="data:image/jpeg;base64,{img_b64}" width="{w}" height="{h}">
  </div>
  <div>
    <div class="sample-meta">{escape(meta.get('shard',''))}/{escape(meta.get('key',''))} · kind={escape(meta.get('kind',''))}</div>
    <div class="section-title">prompt</div><div class="caption-text">{prompt}</div>
    <div class="section-title">responses (GT label = Mark {gt_mark})</div>
    <div class="caption-list">{resp_html}</div>
    <div class="section-title">coords</div>
    <div class="caption-text">gt_norm: ({gt[0]:.3f}, {gt[1]:.3f}) · wrong_norm: ({wrong[0]:.3f}, {wrong[1]:.3f})</div>
  </div>
</div>'''


_CACHE_FILE = '/weka/oe-training-default/zixianm/yinuoy/grounding-viz/.cache/structure.pkl'

def _save_structure(structure):
    import pickle, os as _os
    _os.makedirs(_os.path.dirname(_CACHE_FILE), exist_ok=True)
    # Replace render_fn callable -> str name for picklability
    serial = []
    for entry in structure:
        if len(entry) == 3 and entry[2] is True:  # grouped
            cat, groups, gflag = entry
            new_groups = [(gname, [(dn, s, rf.__name__, m) for (dn, s, rf, m) in dsl]) for gname, dsl in groups]
            serial.append((cat, new_groups, gflag))
        else:
            cat, datasets = entry
            new_ds = [(dn, s, rf.__name__, m) for (dn, s, rf, m) in datasets]
            serial.append((cat, new_ds))
    with open(_CACHE_FILE, 'wb') as f: pickle.dump(serial, f)
    print(f'  [cache] saved {_CACHE_FILE}')

def _load_structure():
    import pickle
    if not os.path.exists(_CACHE_FILE): return None
    if os.path.getmtime(_CACHE_FILE) < os.path.getmtime(__file__): return None
    with open(_CACHE_FILE, 'rb') as f: serial = pickle.load(f)
    g = globals()
    structure = []
    for entry in serial:
        if len(entry) == 3 and entry[2] is True:
            cat, groups, gflag = entry
            new_groups = [(gname, [(dn, s, g[rf], m) for (dn, s, rf, m) in dsl]) for gname, dsl in groups]
            structure.append((cat, new_groups, gflag))
        else:
            cat, datasets = entry
            new_ds = [(dn, s, g[rf], m) for (dn, s, rf, m) in datasets]
            structure.append((cat, new_ds))
    print(f'  [cache] loaded {_CACHE_FILE} (skip resampling). use --force to refresh.')
    return structure

def build():
    random.seed(0)
    # Try cache first (use --force or delete .cache/structure.pkl to refresh)
    if '--force' not in sys.argv:
        cached = _load_structure()
        if cached is not None:
            structure = cached
            return _emit_html(structure)
    structure = []

    cat_grounding = []
    for name, (path, desc, sc, size) in GROUNDING_TAR_DATASETS.items():
        print(f'  [grounding tar] sampling {name}...')
        samples, _ = sample_grounding_tar(path, SAMPLES)
        cat_grounding.append((name, samples, render_grounding_tar_sample, (desc, sc, size, path)))
    for name, (path, desc, sc, size) in WEB_GROUNDING_DATASETS.items():
        print(f'  [web grounding] sampling {name}...')
        samples, _ = sample_web_grounding(path, SAMPLES)
        cat_grounding.append((name, samples, render_web_grounding_sample, (desc, sc, size, path)))
    structure.append(('Grounding', cat_grounding))

        # OCR is grouped: [(group_name, [(ds_name, samples, render, meta), ...])]
    grp_pixmo = []
    for sub, (desc, sc, size) in OCR_SUBSETS.items():
        print(f'  [ocr/pixmo_docs] sampling {sub}...')
        path = os.path.join(OCR_BASE, sub)
        samples, _ = sample_ocr_subset(path, SAMPLES)
        grp_pixmo.append((sub, samples, render_ocr_sample, (desc, sc, size, path)))
    grp_cambrian = []
    for sub, (desc, sc, size) in CAMBRIAN_OCR_SUBSETS.items():
        print(f'  [ocr/cambrian] sampling {sub}...')
        path = os.path.join(CAMBRIAN_OCR_BASE, sub)
        samples, _ = sample_cambrian_subset(path, 30)
        grp_cambrian.append((sub, samples, render_ocr_sample, (desc, sc, size, path)))
    grp_olmocr = []
    for sub, (desc, sc, size, pq_name) in OLMOCR_SUBSETS.items():
        print(f'  [ocr/olmocr] sampling {sub}...')
        samples, _ = sample_olmocr_subset(pq_name, 15)
        grp_olmocr.append((sub, samples, render_olmocr_sample, (desc, sc, size, os.path.join(OLMOCR_BASE, pq_name))))
    print('  [ocr/textocr] sampling textocr...')
    samples, _ = sample_textocr(15)
    desc, sc, size = TEXTOCR_CAPS_SUBSETS['textocr']
    grp_textocr = [('textocr', samples, render_textocr_sample, (desc, sc, size, TEXTOCR_BASE))]
    print('  [ocr/textcaps] sampling textcaps...')
    samples, _ = sample_textcaps(15)
    desc, sc, size = TEXTOCR_CAPS_SUBSETS['textcaps']
    grp_textcaps = [('textcaps', samples, render_textcaps_sample, (desc, sc, size, TEXTCAPS_BASE))]
    cat_ocr = [('text_rich_caption', grp_pixmo), ('Cambrian', grp_cambrian),
               ('OlmoOCR', grp_olmocr), ('OcrText', grp_textocr), ('OcrCaps', grp_textcaps)]
    structure.append(('OCR', cat_ocr, True))  # True => grouped

    cat_know = []
    for name, (path, desc, sc, size) in KNOWLEDGE_DATASETS.items():
        print(f'  [knowledge] sampling {name}...')
        samples, _ = sample_knowledge(path, SAMPLES)
        cat_know.append((name, samples, render_knowledge_sample, (desc, sc, size, path)))
    structure.append(('Knowledge', cat_know))

    # Grounding data filtering — show 4 datasets
    cat_filter = []
    for name, (jsonl_dir, tar_root, desc) in FILTER_DATASETS.items():
        print(f'  [filter] sampling {name}...')
        # 50/50 mix of error/non-error
        n_each = max(1, SAMPLES // 2)
        hard, _ = sample_filter_jsonl(jsonl_dir, tar_root, n_each, want_error=True)
        easy, _ = sample_filter_jsonl(jsonl_dir, tar_root, n_each, want_error=False)
        # Count totals
        total = err = 0
        for f in glob(os.path.join(jsonl_dir, 'worker_*.jsonl')):
            for line in open(f):
                try:
                    r = json.loads(line); total += 1
                    if r.get('error'): err += 1
                except: pass
        sc_str = f'{total:,} samples · {err:,} errors ({100*err/max(total,1):.1f}%)'
        size_str = f'{len(hard) + len(easy)} examples shown'
        all_samples = hard + easy
        random.shuffle(all_samples)
        cat_filter.append((name, all_samples, render_filter_sample, (desc, sc_str, size_str, jsonl_dir)))
    structure.append(('Grounding data filtering', cat_filter))

    # Disabled: legacy SOM RM preview replaced by Grounding data filtering above
    cat_som = []
    if False:
     for name, (jsonl_path, desc, sc, size) in SOM_PREVIEW_DATASETS.items():
        print(f'  [som] sampling {name}...')
        samples, _ = sample_som_jsonl(jsonl_path, SAMPLES)
        cat_som.append((name, samples, render_som_sample, (desc, sc, size, jsonl_path)))
    if cat_som:
        structure.append(('SOM RM', cat_som))

    _save_structure(structure)
    return _emit_html(structure)


def _emit_html(structure):
    parts = ['''<!DOCTYPE html><html><head><meta charset="UTF-8"><title>Dataset Visualizer</title>
<style>
:root{--bg:#FAF2E9;--fg:#0A3235;--accent:#F0529C;--accent2:#105257;
  --muted:rgba(10,50,53,0.55);--line:rgba(10,50,53,0.15);--cream-dark:#F1E4D1;}
body{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;margin:0 auto;padding:24px;background:var(--bg);color:var(--fg);max-width:1400px;}
h1{margin:0 0 8px;color:var(--fg);}
.subtitle{color:var(--muted);margin-bottom:16px;font-size:14px;}
.cat-tabs{display:flex;gap:12px;margin-bottom:20px;border-bottom:3px solid var(--accent);}
.cat-tab{padding:12px 28px;border:none;background:transparent;cursor:pointer;font-size:16px;font-weight:600;color:var(--muted);border-radius:8px 8px 0 0;}
.cat-tab.active{background:var(--accent);color:var(--bg);}
.ds-tabs{display:none;gap:6px;flex-wrap:wrap;margin-bottom:16px;border-bottom:1px solid var(--line);padding-bottom:6px;}
.ds-tabs.active{display:flex;}
.ds-tab{padding:8px 14px;border:1px solid var(--line);background:var(--bg);cursor:pointer;font-size:13px;border-radius:6px;color:var(--fg);}
.ds-tab.active{background:var(--accent2);color:var(--bg);border-color:var(--accent2);}
.grp-tabs{display:none;gap:8px;flex-wrap:wrap;margin-bottom:10px;}
.grp-tabs.active{display:flex;}
.grp-tab{padding:6px 14px;border:1px solid var(--accent);background:var(--bg);cursor:pointer;font-size:13px;font-weight:600;border-radius:14px;color:var(--accent);}
.grp-tab.active{background:var(--accent);color:var(--bg);}
.panel{display:none;}
.panel.active{display:block;}
.dataset-intro{background:var(--cream-dark);padding:14px 18px;border-radius:8px;margin-bottom:16px;color:var(--fg);border-left:4px solid var(--accent2);}
.intro-title{font-size:16px;line-height:1.4;margin-bottom:4px;}
.intro-desc{font-size:13px;line-height:1.5;margin-bottom:6px;}
.intro-meta{font-size:11px;color:var(--muted);font-family:ui-monospace,SFMono-Regular,Menlo,monospace;}
.intro-meta code{background:transparent;font-size:11px;}
.sample{background:var(--bg);border:1px solid var(--line);border-radius:12px;padding:16px;margin-bottom:16px;box-shadow:0 1px 3px rgba(10,50,53,0.08);display:grid;grid-template-columns:420px 1fr;gap:16px;}
.knowledge-sample{grid-template-columns:500px 1fr;}
.img-wrap{position:relative;display:block;}
.img-wrap img{max-width:100%;height:auto;display:block;border-radius:8px;border:1px solid var(--line);}
.overlay{position:absolute;top:0;left:0;width:100%;height:100%;pointer-events:none;}
.overlay .box{pointer-events:auto;cursor:pointer;transition:all .15s;}
.overlay .box.dim{opacity:.2;}
.overlay .box.highlight{stroke-width:4;}
.tooltip{position:absolute;background:var(--fg);color:var(--bg);padding:8px 12px;border-radius:6px;font-size:13px;font-weight:600;max-width:320px;pointer-events:none;z-index:100;display:none;border:2px solid var(--bg);box-shadow:0 4px 12px rgba(10,50,53,0.4);}
.tooltip.visible{display:block;}
.sample-meta{font-size:11px;color:var(--muted);margin-bottom:8px;font-family:ui-monospace,SFMono-Regular,Menlo,monospace;}
.section-title{font-size:11px;color:var(--accent);text-transform:uppercase;margin:10px 0 4px;font-weight:700;letter-spacing:.5px;}
.caption-list{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12px;line-height:1.5;max-height:400px;overflow-y:auto;}
.caption-text{font-size:13px;line-height:1.5;word-break:break-word;color:var(--fg);}
.cap{padding:6px 10px;margin-bottom:4px;background:var(--cream-dark);border-radius:4px;cursor:pointer;transition:all .15s;color:var(--fg);}
.cap:hover,.cap.highlight{background:var(--accent);color:var(--bg);}
.cap.dim{opacity:.3;}
.tag-bbox{color:var(--accent2);font-weight:600;}
.tag-pt{color:var(--accent);font-weight:600;}
.knowledge-img-grid{}
.thumb-grid{display:grid;grid-template-columns:repeat(3,1fr);gap:6px;}
.knowledge-thumb{width:100%;height:140px;border-radius:4px;border:1px solid var(--line);object-fit:cover;background:var(--cream-dark);}
.empty{color:var(--muted);font-style:italic;padding:20px;}
a{color:var(--accent2);} a:hover{color:var(--accent);}
details summary{cursor:pointer;color:var(--accent);font-weight:600;}
</style></head><body>
<h1>Dataset Visualizer</h1>
<div class="subtitle">hover a point/bbox to highlight the matching caption</div>
<div class="cat-tabs">''']
    for i, entry in enumerate(structure):
        cat = entry[0]
        active = ' active' if i == 0 else ''
        parts.append(f'<button class="cat-tab{active}" data-cat="cat{i}">{escape(cat)}</button>')
    parts.append('</div>')

    def render_panel(panel_id, cat_attr, dname, samples, render_fn, meta, sid_start, active):
        desc, sc, size, path = meta
        active_cls = ' active' if active else ''
        out = [f'<div id="{panel_id}" class="panel{active_cls}" data-cat="{cat_attr}">']
        out.append(
            f'<div class="dataset-intro">'
            f'<div class="intro-title"><b>{escape(dname)}</b> &nbsp;{escape(sc)} samples</div>'
            f'<div class="intro-desc">{escape(desc)}</div>'
            f'<div class="intro-meta">{escape(size)} · <code>{escape(path)}</code></div>'
            f'</div>'
        )
        sid = sid_start
        if not samples:
            out.append(f'<div class="empty">No samples found in {escape(dname)}</div>')
        else:
            out.append(f'<div class="sample-meta" style="margin-bottom:12px"><b>{len(samples)} random samples shown</b></div>')
            for s in samples:
                out.append(render_fn(s, sid))
                sid += 1
        out.append('</div>')
        return '\n'.join(out), sid

    global_sid = 0
    for i, entry in enumerate(structure):
        is_grouped = (len(entry) == 3 and entry[2] is True)
        cat = entry[0]
        groups_or_datasets = entry[1]
        active_cat = ' active' if i == 0 else ''
        if is_grouped:
            # Render group-tabs row (always present for grouped cats)
            parts.append(f'<div id="cat{i}_grps" class="grp-tabs{active_cat}">')
            for j, (gname, _) in enumerate(groups_or_datasets):
                active_g = ' active' if j == 0 else ''
                parts.append(f'<button class="grp-tab{active_g}" data-cat="cat{i}" data-grp="g_{i}_{j}">{escape(gname)}</button>')
            parts.append('</div>')
            # Render one ds-tabs row per group; only the first group active
            for j, (gname, datasets) in enumerate(groups_or_datasets):
                active_dst = ' active' if (i == 0 and j == 0) else ''
                parts.append(f'<div id="g_{i}_{j}_tabs" class="ds-tabs{active_dst}" data-grp-of="cat{i}">')
                for k, (dname, _, _, _) in enumerate(datasets):
                    active_ds = ' active' if k == 0 else ''
                    parts.append(f'<button class="ds-tab{active_ds}" data-panel="p_{i}_{j}_{k}">{escape(dname)}</button>')
                parts.append('</div>')
            # Render panels
            for j, (gname, datasets) in enumerate(groups_or_datasets):
                for k, (dname, samples, render_fn, meta) in enumerate(datasets):
                    pid = f'p_{i}_{j}_{k}'
                    active_p = (i == 0 and j == 0 and k == 0)
                    panel_html, global_sid = render_panel(pid, f'cat{i}', dname, samples, render_fn, meta, global_sid, active_p)
                    parts.append(panel_html)
        else:
            # Flat cat (backward compat): single ds-tabs row
            datasets = groups_or_datasets
            parts.append(f'<div id="cat{i}_tabs" class="ds-tabs{active_cat}">')
            for j, (dname, _, _, _) in enumerate(datasets):
                active_ds = ' active' if j == 0 else ''
                parts.append(f'<button class="ds-tab{active_ds}" data-panel="p_{i}_{j}">{escape(dname)}</button>')
            parts.append('</div>')
            for j, (dname, samples, render_fn, meta) in enumerate(datasets):
                pid = f'p_{i}_{j}'
                active_p = (i == 0 and j == 0)
                panel_html, global_sid = render_panel(pid, f'cat{i}', dname, samples, render_fn, meta, global_sid, active_p)
                parts.append(panel_html)

    parts.append('''<script>
function highlightSample(sid, ai) {
  const bs = document.querySelectorAll('.box.s' + sid);
  const cs = document.querySelectorAll('.sample[data-sample-idx="' + sid + '"] .cap');
  if (ai === null) { bs.forEach(b => b.classList.remove('highlight','dim')); cs.forEach(c => c.classList.remove('highlight','dim')); return; }
  bs.forEach(b => { if (b.dataset.idx == ai) { b.classList.add('highlight'); b.classList.remove('dim'); } else { b.classList.add('dim'); b.classList.remove('highlight'); } });
  cs.forEach(c => { if (c.dataset.idx == ai) { c.classList.add('highlight'); c.classList.remove('dim'); } else { c.classList.add('dim'); c.classList.remove('highlight'); } });
}
document.querySelectorAll('.box').forEach(b => {
  const sid = [...b.classList].find(c => c.startsWith('s') && c !== 'box').slice(1);
  const sampleEl = b.closest('.sample'); const tooltip = sampleEl.querySelector('.tooltip'); const imgWrap = sampleEl.querySelector('.img-wrap');
  b.addEventListener('mouseenter', () => { highlightSample(sid, b.dataset.idx); if (tooltip) { tooltip.textContent = b.dataset.phrase; tooltip.classList.add('visible'); } });
  b.addEventListener('mousemove', (e) => { if (tooltip) { const r = imgWrap.getBoundingClientRect(); tooltip.style.left = (e.clientX - r.left + 12) + 'px'; tooltip.style.top = (e.clientY - r.top + 12) + 'px'; } });
  b.addEventListener('mouseleave', () => { highlightSample(sid, null); if (tooltip) tooltip.classList.remove('visible'); });
});
document.querySelectorAll('.cap').forEach(c => {
  c.addEventListener('mouseenter', () => highlightSample(c.dataset.sample, c.dataset.idx));
  c.addEventListener('mouseleave', () => highlightSample(c.dataset.sample, null));
});
document.querySelectorAll('.cat-tab').forEach(t => {
  t.addEventListener('click', () => {
    document.querySelectorAll('.cat-tab').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.grp-tabs').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.grp-tab').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.ds-tabs').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.ds-tab').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.panel').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    const cat = t.dataset.cat;
    const grps = document.getElementById(cat + '_grps');
    if (grps) {
      grps.classList.add('active');
      const firstGrp = grps.querySelector('.grp-tab');
      if (firstGrp) {
        firstGrp.classList.add('active');
        const grpTabs = document.getElementById(firstGrp.dataset.grp + '_tabs');
        if (grpTabs) {
          grpTabs.classList.add('active');
          const firstDs = grpTabs.querySelector('.ds-tab');
          if (firstDs) { firstDs.classList.add('active'); document.getElementById(firstDs.dataset.panel).classList.add('active'); }
        }
      }
    } else {
      const tabs = document.getElementById(cat + '_tabs'); if (tabs) tabs.classList.add('active');
      const firstDs = tabs && tabs.querySelector('.ds-tab');
      if (firstDs) { firstDs.classList.add('active'); document.getElementById(firstDs.dataset.panel).classList.add('active'); }
    }
  });
});
document.querySelectorAll('.grp-tab').forEach(t => {
  t.addEventListener('click', () => {
    const cat = t.dataset.cat;
    // Hide all ds-tabs and panels for this cat
    document.querySelectorAll('.ds-tabs[data-grp-of="' + cat + '"]').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.panel[data-cat="' + cat + '"]').forEach(x => x.classList.remove('active'));
    document.querySelectorAll('.grp-tab[data-cat="' + cat + '"]').forEach(x => x.classList.remove('active'));
    t.classList.add('active');
    const grpTabs = document.getElementById(t.dataset.grp + '_tabs');
    if (grpTabs) {
      grpTabs.classList.add('active');
      grpTabs.querySelectorAll('.ds-tab').forEach(x => x.classList.remove('active'));
      const firstDs = grpTabs.querySelector('.ds-tab');
      if (firstDs) { firstDs.classList.add('active'); document.getElementById(firstDs.dataset.panel).classList.add('active'); }
    }
  });
});
document.querySelectorAll('.ds-tab').forEach(t => {
  t.addEventListener('click', () => {
    const cat = t.closest('.ds-tabs'); cat.querySelectorAll('.ds-tab').forEach(x => x.classList.remove('active')); t.classList.add('active');
    const panel = document.getElementById(t.dataset.panel);
    document.querySelectorAll('.panel[data-cat="' + panel.dataset.cat + '"]').forEach(p => p.classList.remove('active'));
    panel.classList.add('active');
  });
});
</script></body></html>''')

    html = '\n'.join(parts)

    # Inject preserved panels that aren't (yet) generated by this script:
    #   cat0 RF100 + ScreenSpot (Grounding)
    #   cat2 InfoSeek + OVEN-entity + OVEN-query (Knowledge)
    # Snippets are manually curated HTML with pre-extracted sample images
    # (source data is in large tars, slow to read each regen).
    import re as _re
    snip_dir = os.path.join(os.path.dirname(OUT), '.snippets')
    def _inject(html, tabs_file, panels_file, tabs_id_re, panels_anchor_re, label):
        tp = os.path.join(snip_dir, tabs_file)
        pp = os.path.join(snip_dir, panels_file)
        if not (os.path.isfile(tp) and os.path.isfile(pp)): return html
        html = _re.sub(tabs_id_re, open(tp).read(), html, count=1, flags=_re.DOTALL)
        m = _re.search(panels_anchor_re, html)
        if m:
            inject_at = m.start()
            html = html[:inject_at] + '\n' + open(pp).read() + '\n' + html[inject_at:]
            print(f'  [snippet] injected {label}')
        return html

    html = _inject(html, 'cat0_tabs.html', 'cat0_rf100_seeclick_panels.html',
                   r'<div id="cat0_tabs"[^>]*>.*?</div>',
                   r'<div id="cat1_(?:tabs|grps)"', 'RF100/ScreenSpot under Grounding')
    html = _inject(html, 'cat2_tabs.html', 'cat2_oven_infoseek_panels.html',
                   r'<div id="cat2_tabs"[^>]*>.*?</div>',
                   r'<div id="p_3_0"', 'InfoSeek/OVEN under Knowledge')

    # cat3: REPLACE generate.py's default 4-tab + 4-panel cat3 with snippet-based content
    # from scripts/build_cat3_dataset_panels.py + scripts/render_judge_ensemble.py.
    def _cat3_inject(html):
        tabs_ds = os.path.join(snip_dir, 'cat3_dataset_tabs.html')
        tabs_jg = os.path.join(snip_dir, 'cat3_judge_tab.html')
        panels_ds = os.path.join(snip_dir, 'cat3_dataset_panels.html')
        panel_jg = os.path.join(snip_dir, 'cat3_judge_panel.html')
        if not os.path.isfile(tabs_ds): return html
        new_tabs = open(tabs_ds).read().rstrip()
        if os.path.isfile(tabs_jg):
            new_tabs = _re.sub(r'</div>\s*$', open(tabs_jg).read() + '</div>', new_tabs)
        new_panels = (open(panels_ds).read() if os.path.isfile(panels_ds) else '')
        if os.path.isfile(panel_jg):
            new_panels += '\n' + open(panel_jg).read()
        html = _re.sub(r'<div id="cat3_tabs"[^>]*>.*?</div>', new_tabs, html, count=1, flags=_re.DOTALL)
        for pid in ('p_3_0','p_3_1','p_3_2','p_3_3'):
            html = _re.sub(rf'<div id="{pid}" class="panel" data-cat="cat3">.*?</div>(?=\s*<div id="p_[03]_|\s*</body>|\s*<script>)',
                           '', html, count=1, flags=_re.DOTALL)
        html = html.replace('</body>', new_panels + '\n</body>', 1)
        print(f'  [snippet] injected cat3 (dataset + judge ensemble) — {len(new_panels):,} bytes panels')
        return html
    html = _cat3_inject(html)

    with open(OUT, 'w') as f: f.write(html)
    sz = os.path.getsize(OUT)/1024/1024
    print(f'\nWrote {OUT} ({sz:.1f} MB)')


if __name__ == '__main__':
    build()
