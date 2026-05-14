"""Add RF100 panel to grounding-viz/index.html.
- Pick 1 random dataset per RF100 category (7 total)
- Extract 1 sample image + JSON from packed tar
- Build SVG bbox overlay + caption list
- Insert as new `RF100` ds-tab in Grounding (cat0) with 7 category sub-tabs
"""
import base64
import csv
import io
import json
import os
import random
import re
import tarfile
from collections import defaultdict
from pathlib import Path

import PIL.Image

ROOT = Path("/weka/oe-training-default/zixianm/yinuoy")
INDEX = ROOT / "grounding-viz" / "index.html"
TARS = Path("/weka/oe-training-default/oe-encoder/roboflow100_tars")
CATS_CSV = "/tmp/rf100_cats.csv"
COLORS = ["#e53e3e", "#38a169", "#3182ce", "#d69e2e", "#805ad5", "#dd6b20", "#319795", "#d53f8c", "#2b6cb0", "#22543d"]

random.seed(42)


def load_categories():
    cat_to_datasets = defaultdict(list)
    with open(CATS_CSV) as f:
        rdr = csv.reader(f)
        next(rdr)
        for ds, cat in rdr:
            cat_to_datasets[cat.strip()].append(ds.strip())
    return cat_to_datasets


def pick_sample_from_tar(slug):
    """Return (img_b64, captions_list, w, h) for 1 random sample."""
    tar_dir = TARS / slug
    tars = sorted(tar_dir.glob("*.tar"))
    if not tars:
        return None
    tar_path = tars[0]
    with tarfile.open(tar_path) as t:
        members = t.getmembers()
        stems = sorted(set(m.name.rsplit('.', 1)[0] for m in members))
        if not stems:
            return None
        # try a few; some may fail to decode
        for stem in random.sample(stems, min(10, len(stems))):
            try:
                jpg = t.extractfile(f"{stem}.jpg")
                if jpg is None: continue
                img_bytes = jpg.read()
                meta_f = t.extractfile(f"{stem}.json")
                if meta_f is None: continue
                meta = json.loads(meta_f.read())
                img = PIL.Image.open(io.BytesIO(img_bytes))
                W, H = img.size
                MAX_W = 400  # smaller resize to keep HTML small with multiple samples per cat
                if W > MAX_W:
                    new_h = int(H * MAX_W / W)
                    img = img.resize((MAX_W, new_h))
                    W, H = MAX_W, new_h
                    buf = io.BytesIO()
                    img.convert("RGB").save(buf, "JPEG", quality=80)
                    img_bytes = buf.getvalue()
                b64 = base64.b64encode(img_bytes).decode()
                return b64, meta["captions"], W, H
            except Exception:
                continue
    return None


def parse_v3_caption(cap_str):
    """parse 'class (box) ; x1,y1,x2,y2 ; ...' -> (class_name, [(x1,y1,x2,y2), ...])"""
    if "(box) ;" not in cap_str:
        return None, []
    head, rest = cap_str.split(" (box) ; ", 1)
    boxes = []
    for chunk in rest.split(" ; "):
        try:
            parts = [int(x.strip()) for x in chunk.split(",")]
            if len(parts) == 4:
                boxes.append(parts)
        except ValueError:
            pass
    return head.strip(), boxes


def build_panel(slug, panel_id, img_b64, captions, W, H):
    """Build HTML for one sub-panel showing the RF100 sample."""
    # Render image + svg overlays
    parts = []
    parts.append(f'<div class="sample" data-sample-idx="{panel_id}" style="margin-bottom:20px">')
    parts.append(f'<div style="font-weight:600;margin-bottom:6px"><code>{slug}</code></div>')
    parts.append(f'<div class="img-wrap" style="width:{W}px;height:{H}px;position:relative">')
    parts.append(f'<img src="data:image/jpeg;base64,{img_b64}" style="width:100%;height:100%;display:block;border:1px solid var(--line);border-radius:4px;">')
    # SVG overlay (viewBox in 0-999 native grid)
    parts.append(f'<svg class="overlay" viewBox="0 0 999 999" preserveAspectRatio="none" style="position:absolute;top:0;left:0;width:100%;height:100%">')
    cap_render = []
    for ci, cap_str in enumerate(captions):
        cname, boxes = parse_v3_caption(cap_str)
        if cname is None or not boxes:
            continue
        color = COLORS[ci % len(COLORS)]
        for bi, (x1, y1, x2, y2) in enumerate(boxes):
            # Use 'rf-box' class so existing JS .querySelectorAll('.box') doesn't pick these up.
            # data-idx links to the matching rf-cap; data-phrase for tooltip.
            parts.append(
                f'<rect class="rf-box" data-idx="{ci}" data-phrase="{cname}" '
                f'x="{x1}" y="{y1}" width="{x2-x1}" height="{y2-y1}" '
                f'stroke="{color}" fill="transparent" stroke-width="3" style="cursor:pointer"></rect>'
            )
        cap_render.append((ci, cname, color, len(boxes)))
    parts.append('</svg></div>')
    parts.append(f'<div class="section-title" style="margin-top:10px">classes ({len(cap_render)})</div>')
    parts.append('<div class="caption-list">')
    for ci, cname, color, nbox in cap_render:
        parts.append(
            f'<div class="rf-cap" data-idx="{ci}" style="border-left: 3px solid {color};padding:6px 10px;margin-bottom:4px;background:var(--cream-dark);border-radius:4px;color:var(--fg);cursor:pointer;transition:all .15s;">'
            f'<b>{cname}</b> <span class="tag-bbox">({nbox} bbox{"es" if nbox>1 else ""})</span></div>'
        )
    parts.append('</div></div>')
    return "".join(parts)


def build_rf100_full_panel(cat_to_datasets, samples_per_cat: int = 5):
    """Build the full p_0_9 panel with sub-tabs for 7 categories,
    showing up to `samples_per_cat` distinct datasets per category."""
    cats = sorted(cat_to_datasets.keys())
    panel_html = []
    panel_html.append('<div class="dataset-intro"><div class="intro-title"><b>RF100 (Roboflow-100)</b> &nbsp;~225K samples · 100 datasets · 7 categories</div>'
                     f'<div class="intro-desc">Diverse bbox grounding benchmark covering aerial, documents, electromagnetic, microscopic, real-world, underwater, video-games domains. '
                     f'Up to {samples_per_cat} random datasets shown per category.</div>'
                     '<div class="intro-meta">36 GB · <code>/weka/oe-training-default/oe-encoder/roboflow100_tars</code></div></div>')
    panel_html.append('<div class="sub-tabs">')
    for ci, cat in enumerate(cats):
        active = " active" if ci == 0 else ""
        panel_html.append(f'<button class="sub-tab{active}" data-sub="sub_rf100_{ci}">{cat} ({len(cat_to_datasets[cat])})</button>')
    panel_html.append('</div>')
    sample_idx = 0  # global counter for unique data-sample-idx across all sub-panels
    for ci, cat in enumerate(cats):
        active = " active" if ci == 0 else ""
        panel_html.append(f'<div data-sub="sub_rf100_{ci}" class="sub-panel{active}">')
        slugs = list(cat_to_datasets[cat])
        random.shuffle(slugs)
        found_count = 0
        for slug in slugs:
            if found_count >= samples_per_cat:
                break
            sample = pick_sample_from_tar(slug)
            if sample is None:
                continue
            img_b64, captions, W, H = sample
            panel_html.append(build_panel(slug, f"rf100_{sample_idx}", img_b64, captions, W, H))
            sample_idx += 1
            found_count += 1
        if found_count == 0:
            panel_html.append(f'<div class="empty">no usable sample found for {cat}</div>')
        panel_html.append('</div>')
    return "".join(panel_html)


def main():
    cat_to_datasets = load_categories()
    print(f"Loaded {sum(len(v) for v in cat_to_datasets.values())} datasets in {len(cat_to_datasets)} categories")

    rf100_panel_inner = build_rf100_full_panel(cat_to_datasets)

    with open(INDEX) as f:
        html = f.read()

    # 1. Add tab button in cat0_tabs (after p_0_8 yfcc15m)
    new_button = '<button class="ds-tab" data-panel="p_0_9">RF100</button>'
    html = html.replace(
        '<button class="ds-tab" data-panel="p_0_8">yfcc15m</button>',
        '<button class="ds-tab" data-panel="p_0_8">yfcc15m</button>\n' + new_button
    )

    # 2. Insert panel before <div id="cat1_tabs"
    new_panel = f'<div id="p_0_9" class="panel" data-cat="cat0">\n{rf100_panel_inner}\n</div>\n'
    html = re.sub(r'(<div id="cat1_tabs")', new_panel + r'\1', html, count=1)

    # 3. Append RF100-specific hover JS just before </body> (doesn't touch existing handlers)
    rf100_js = """
<script>
// RF100 hover-highlight (scoped to .sample, mirrors original .box/.cap behavior)
document.querySelectorAll('.rf-box').forEach(b => {
  const sample = b.closest('.sample');
  if (!sample) return;
  b.addEventListener('mouseenter', () => {
    const idx = b.dataset.idx;
    sample.querySelectorAll('.rf-box').forEach(x => {
      if (x.dataset.idx === idx) { x.style.opacity = '1'; x.setAttribute('stroke-width','5'); }
      else { x.style.opacity = '0.25'; }
    });
    sample.querySelectorAll('.rf-cap').forEach(c => {
      if (c.dataset.idx === idx) { c.style.background = 'var(--accent)'; c.style.color = 'var(--bg)'; }
      else { c.style.opacity = '0.3'; }
    });
  });
  b.addEventListener('mouseleave', () => {
    sample.querySelectorAll('.rf-box').forEach(x => { x.style.opacity = ''; x.setAttribute('stroke-width','3'); });
    sample.querySelectorAll('.rf-cap').forEach(c => { c.style.background = 'var(--cream-dark)'; c.style.color = 'var(--fg)'; c.style.opacity = ''; });
  });
});
document.querySelectorAll('.rf-cap').forEach(c => {
  const sample = c.closest('.sample');
  if (!sample) return;
  c.addEventListener('mouseenter', () => {
    const idx = c.dataset.idx;
    sample.querySelectorAll('.rf-box').forEach(x => {
      if (x.dataset.idx === idx) { x.style.opacity = '1'; x.setAttribute('stroke-width','5'); }
      else { x.style.opacity = '0.25'; }
    });
    sample.querySelectorAll('.rf-cap').forEach(x => {
      if (x === c) { x.style.background = 'var(--accent)'; x.style.color = 'var(--bg)'; }
      else { x.style.opacity = '0.3'; }
    });
  });
  c.addEventListener('mouseleave', () => {
    sample.querySelectorAll('.rf-box').forEach(x => { x.style.opacity = ''; x.setAttribute('stroke-width','3'); });
    sample.querySelectorAll('.rf-cap').forEach(x => { x.style.background = 'var(--cream-dark)'; x.style.color = 'var(--fg)'; x.style.opacity = ''; });
  });
});
</script>
"""
    html = html.replace("</body>", rf100_js + "</body>")

    # backup + write
    bak = INDEX.with_suffix(".html.bak_pre_rf100")
    bak.write_text(open(INDEX).read())
    INDEX.write_text(html)
    print(f"Updated {INDEX} (backup at {bak})")


if __name__ == "__main__":
    main()
