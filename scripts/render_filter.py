"""Render F1<1 cases from grounding_filter_molmo2/{vg,openimages,grit,pixmo}_filtered/.

For each rejected sample: extract original image from webdataset tar, draw red dots at
Molmo's predicted pixel coords, save as annotated/{key}_p00000.jpg.

No GT overlay — the judge is asked simply "do the red dots point at <phrase>?"
"""
import argparse
import glob
import io
import json
import os
import re
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed

from PIL import Image, ImageDraw

TAR_ROOTS = {
    'vg':          '/weka/oe-training-default/oe-encoder/vg',
    'openimages':  '/weka/oe-training-default/oe-encoder/openimages',
    'grit':        '/weka/oe-training-default/oe-encoder/grit',
    'pixmo':       '/weka/oe-training-default/oe-encoder/pixmo',
}
FILTER_ROOT = '/weka/oe-training-default/oe-encoder/grounding_filter_molmo2'

COORD_RE = re.compile(r'coords="(\d+)\s+(\d+)\s+(\d+)\s+(\d+)"')


def resolve_tar(ds, shard):
    """grit uses nested coyo_*_snappy dirs; others flat."""
    direct = os.path.join(TAR_ROOTS[ds], shard)
    if os.path.isfile(direct):
        return direct
    # nested search
    matches = glob.glob(os.path.join(TAR_ROOTS[ds], '**', shard), recursive=True)
    return matches[0] if matches else None


def recover_image_size(pred_pts, pred_raw):
    """Molmo emits `<points coords="x1 y1 x2 y2">`. The 4 values are 0-999 normalized of the
    original image. Pred pixel coords correspond to the same image at native size — recover
    it via the smallest abs scale factor between coords and pixel preds.

    Fallback: if we can't recover from raw, return None and caller should re-decode the image."""
    if not pred_raw or not pred_pts:
        return None, None
    m = COORD_RE.search(pred_raw)
    if not m:
        return None, None
    # Coords from raw are 0-999 normalized. pred_pts are pixel.
    # We can't recover image size purely from this. Just decode image and use its real size.
    return None, None


def extract_image_from_tar(tar_path, key):
    """A webdataset tar contains files like {key}.jpg, {key}.json, etc. Return the JPEG bytes."""
    with tarfile.open(tar_path, 'r') as tf:
        for name in (f'{key}.jpg', f'{key}.jpeg', f'{key}.png', f'{key}.webp'):
            try:
                m = tf.getmember(name)
                return tf.extractfile(m).read()
            except KeyError:
                continue
        # fallback: scan
        for m in tf.getmembers():
            if m.name.startswith(f'{key}.') and m.name.rsplit('.', 1)[-1].lower() in ('jpg','jpeg','png','webp'):
                return tf.extractfile(m).read()
    return None


def _xy_bbox(bb, gt_order):
    """Normalize a GT bbox to [x1,y1,x2,y2]. gt_order 'yx' means stored [y1,x1,y2,x2]."""
    if gt_order == 'yx':
        return [bb[1], bb[0], bb[3], bb[2]]
    return list(bb[:4])


def _xy_point(pt, gt_order):
    """Normalize a GT point to (x,y). gt_order 'yx' means stored (y,x)."""
    if gt_order == 'yx':
        return pt[1], pt[0]
    return pt[0], pt[1]


def render_yellow_red(img_bytes, pred_pts, gt_bboxes=None, gt_points=None,
                       gt_norm=False, pred_norm=False, max_side=1024,
                       radius=6, yellow=(255, 220, 0), red=(220, 30, 60),
                       yellow_alpha=80, gt_order='xy'):
    """Render judge-style image: YELLOW translucent GT region + RED pred dots.

    For bbox GT: translucent yellow fill + solid yellow border.
    For point GT: solid yellow filled circle.
    For pred: solid red filled circle.

    gt_order: 'xy' if GT stored [x1,y1,x2,y2]/(x,y); 'yx' if [y1,x1,y2,x2]/(y,x)
              (the Molmo row-major convention — vg/openimages/pixmo).
    Used as input to 2-stage VLM judge (mask_verdict + molmo_verdict).
    """
    from PIL import ImageDraw
    base = Image.open(io.BytesIO(img_bytes)).convert('RGBA')
    w, h = base.size
    s = 1.0
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        base = base.resize((int(w * s), int(h * s)))
    overlay = Image.new('RGBA', base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    OW, OH = w, h

    def _scale(x, y, norm):
        if norm: x, y = x * OW / 1000.0, y * OH / 1000.0
        return x * s, y * s

    if gt_bboxes:
        for bb in gt_bboxes:
            bb = _xy_bbox(bb, gt_order)
            x1, y1 = _scale(bb[0], bb[1], gt_norm)
            x2, y2 = _scale(bb[2], bb[3], gt_norm)
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            draw.rectangle([x1, y1, x2, y2], fill=(*yellow, yellow_alpha),
                           outline=(*yellow, 255), width=2)
    if gt_points:
        for pt in gt_points:
            gx, gy = _xy_point(pt, gt_order)
            x, y = _scale(gx, gy, gt_norm)
            r = radius + 1
            draw.ellipse([x - r, y - r, x + r, y + r],
                         fill=(*yellow, 220), outline=(0, 0, 0, 255), width=2)
    for (px, py) in pred_pts:
        x, y = _scale(px, py, pred_norm)
        draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                     fill=(*red, 255), outline=(255, 255, 255, 255), width=2)
    out = Image.alpha_composite(base, overlay).convert('RGB')
    return out


def draw_red_dots(img_bytes, pred_pts, gt_bboxes=None, gt_points=None,
                  radius=5, max_side=1024, gt_norm=None, pred_norm=False,
                  gt_order='xy'):
    """Red dot for each Molmo pred; green dot/box for each GT.

    gt_norm: explicit per-dataset flag — True if GT coords are 0-999 normalized,
             False if pixel. Required when gt_bboxes/gt_points is set. (No more
             auto-detect, which was unreliable on large images.)
    gt_order: 'xy' if GT stored [x1,y1,x2,y2]/(x,y); 'yx' if [y1,x1,y2,x2]/(y,x).
    pred_norm: True if Molmo pred_pts are 0-999 normalized (NEVER for our filter
               pipeline, which always converts pred to pixel before storing).
    """
    im = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    w, h = im.size
    s = 1.0
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        im = im.resize((int(w * s), int(h * s)))
    draw = ImageDraw.Draw(im)
    OW, OH = w, h
    GREEN = (22, 163, 74)
    has_gt = gt_bboxes or gt_points
    if has_gt and gt_norm is None:
        raise ValueError('gt_norm flag required when gt_bboxes/gt_points is set (no auto-detect)')

    def _scale(x, y, norm):
        if norm:
            x = x * OW / 1000.0; y = y * OH / 1000.0
        return x * s, y * s

    if gt_bboxes:
        for bb in gt_bboxes:
            bb = _xy_bbox(bb, gt_order)
            x1, y1 = _scale(bb[0], bb[1], gt_norm)
            x2, y2 = _scale(bb[2], bb[3], gt_norm)
            x1, x2 = min(x1, x2), max(x1, x2)
            y1, y2 = min(y1, y2), max(y1, y2)
            draw.rectangle([x1, y1, x2, y2], outline=GREEN, width=2)
            cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
            draw.ellipse([cx - radius, cy - radius, cx + radius, cy + radius],
                         fill=GREEN, outline=(255, 255, 255), width=2)
    if gt_points:
        for pt in gt_points:
            gx, gy = _xy_point(pt, gt_order)
            x, y = _scale(gx, gy, gt_norm)
            draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                         fill=GREEN, outline=(255, 255, 255), width=2)
    for (px, py) in pred_pts:
        x, y = _scale(px, py, pred_norm)
        draw.ellipse([x - radius, y - radius, x + radius, y + radius],
                     fill=(220, 30, 60), outline=(255, 255, 255), width=2)
    return im


# Per-dataset GT coord system (True = 0-999 normalized, False = pixel)
GT_NORM_BY_DS = {
    'vg':         True,   # 0-999 normalized (verified May 7 vg_bbox_debug.html)
    'openimages': True,   # max values cluster at 998 → normalized
    'pixmo':      True,   # max values 940-980 → normalized
    'grit':       False,  # legacy grit (pre-shard-fix)
    'grit_v2':    False,  # pixel: image is 1024x1024 with gt up to 1023
    'cc3m':       False,  # pixel
    'cc12m':      False,  # pixel
    'seeclick':   False,  # pixel
    'rf100':      False,  # pixel
}

# Per-dataset GT coordinate ORDER. 'yx' = Molmo row-major [y1,x1,y2,x2]/(y,x);
# 'xy' = standard [x1,y1,x2,y2]/(x,y).  All CONFIRMED May 20:
#   vg/openimages/pixmo 'yx' — visual check (red shirt / cake / telephone) + bounds.
#   grit_v2/cc3m/cc12m/rf100/seeclick 'xy' — statistical bounds check (coord0 ≤ W always).
GT_ORDER_BY_DS = {
    'vg':         'yx',
    'openimages': 'yx',
    'pixmo':      'yx',
    'grit':       'xy',
    'grit_v2':    'xy',
    'cc3m':       'xy',
    'cc12m':      'xy',
    'seeclick':   'xy',
    'rf100':      'xy',
}


def iter_filter_records(ds, max_samples=None):
    """Yield F1<1 records from ds_filtered/."""
    pat = os.path.join(FILTER_ROOT, f'{ds}_filtered', 'worker_*.jsonl')
    n = 0
    for jf in sorted(glob.glob(pat)):
        for line in open(jf):
            try:
                r = json.loads(line)
            except Exception:
                continue
            info = r.get('info') or {}
            f1 = info.get('f1', 1.0)
            if f1 >= 1.0 - 1e-9:
                continue
            yield r
            n += 1
            if max_samples and n >= max_samples:
                return


def _render_one(args_tup):
    """Worker function: render one sample. Returns manifest dict or None on skip."""
    ds, rec, out_dir = args_tup
    shard = rec['shard']; key = rec['key']; phrase = rec.get('phrase', '')
    preds = rec.get('pred') or []
    if not preds:
        return None
    tar_path = resolve_tar(ds, shard)
    if tar_path is None:
        return None
    img_bytes = extract_image_from_tar(tar_path, key)
    if img_bytes is None:
        return None
    try:
        # Pass GT (bboxes for VG/OI/GRIT, points for Pixmo)
        gt = rec.get('gt') or []
        kind = rec.get('kind', 'bbox')
        gt_bboxes = gt if kind == 'bbox' else None
        gt_points = gt if kind == 'point' else None
        im = draw_red_dots(img_bytes, preds, gt_bboxes=gt_bboxes, gt_points=gt_points,
                           gt_norm=GT_NORM_BY_DS.get(ds, False),
                           gt_order=GT_ORDER_BY_DS.get(ds, 'xy'))
        out_jpg = os.path.join(out_dir, f'{key}_p00000.jpg')
        im.save(out_jpg, 'JPEG', quality=82)
    except Exception:
        return None
    info = rec.get('info') or {}
    return {
        'dataset': ds,
        'key': key,
        'pair_idx': 0,
        'phrase': phrase,
        'n_pred': info.get('n_pred', len(preds)),
        'n_gt': info.get('n_gt', len(rec.get('gt') or [])),
        'f1': info.get('f1', 0.0),
        'kind': rec.get('kind'),
        'img_path': out_jpg,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--datasets', nargs='+', default=['vg', 'openimages', 'grit', 'pixmo'])
    ap.add_argument('--per_ds', type=int, default=50, help='F1<1 samples per dataset')
    ap.add_argument('--out_root', default=FILTER_ROOT)
    ap.add_argument('--manifest', default=None)
    ap.add_argument('--workers', type=int, default=8, help='Parallel render workers (ProcessPool)')
    ap.add_argument('--log_every', type=int, default=200)
    args = ap.parse_args()

    manifest_tmp = (args.manifest + '.tmp') if args.manifest else None
    manifest_fh = open(manifest_tmp, 'w') if manifest_tmp else None

    import time
    t0 = time.time()
    for ds in args.datasets:
        out_dir = os.path.join(args.out_root, f'{ds}_filtered', 'annotated_redonly')
        os.makedirs(out_dir, exist_ok=True)
        records = list(iter_filter_records(ds, max_samples=args.per_ds))
        print(f'{ds}: queued {len(records)} F1<1 records, rendering with {args.workers} workers...', flush=True)
        n_done = n_skip = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = [ex.submit(_render_one, (ds, r, out_dir)) for r in records]
            for fu in as_completed(futures):
                res = fu.result()
                if res is None:
                    n_skip += 1
                else:
                    n_done += 1
                    if manifest_fh:
                        manifest_fh.write(json.dumps(res) + '\n')
                        manifest_fh.flush()
                tot = n_done + n_skip
                if tot % args.log_every == 0:
                    el = time.time() - t0
                    rate = tot / max(el, 1)
                    print(f'  [{ds}] {tot}/{len(records)} done={n_done} skip={n_skip} · {rate:.1f}/s', flush=True)
        print(f'{ds}: rendered={n_done} skipped={n_skip} → {out_dir}', flush=True)

    if manifest_fh:
        manifest_fh.close()
        os.rename(manifest_tmp, args.manifest)
        print(f'manifest → {args.manifest}', flush=True)
    print(f'TOTAL elapsed: {time.time() - t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
