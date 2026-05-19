"""Build sharded judge manifest for ALL F1<1 rejected records of one dataset.

Each shard produces:
  - manifest-shard-NNN-of-MMM.jsonl
  - imgs/{ds}_{uniq_key}.jpg (one per row)

Reads filter records via load_rejected from build_cat3_dataset_panels (handles both legacy + new schema).
Uses render_yellow_red to render image with YELLOW GT + RED pred for 2-stage judge prompt.

Usage:
  python build_judge_manifest_full.py --ds vg --out_dir /weka/.../judge_full/vg --shard 0/8
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_filter import render_yellow_red, extract_image_from_tar, GT_NORM_BY_DS
from build_cat3_dataset_panels import resolve_tar, load_rejected, extract_for_ds

DS_TO_SUB_SCHEMA = {
    'vg':         ('vg_filtered',         'legacy'),
    'openimages': ('openimages_filtered', 'legacy'),
    'pixmo':      ('pixmo_filtered',      'legacy'),
    'grit_v2':    ('grit_v2_filtered',    'new'),
    'cc3m':       ('cc3m_filtered',       'new'),
    'cc12m':      ('cc12m_filtered',      'new'),
    'seeclick':   ('seeclick_filtered',   'new'),
    'rf100':      ('rf100_filtered',      'new'),
}
DS_TO_TAR_DS = {'vg':'vg','openimages':'openimages','pixmo':'pixmo','grit_v2':'grit','cc3m':'cc3m','cc12m':'cc12m','seeclick':'seeclick','rf100':'rf100'}


def safe(s):
    """Filename-safe transform: replace anything outside [A-Za-z0-9_.-] with underscore."""
    import re
    return re.sub(r'[^A-Za-z0-9_.-]', '_', s or '')


def make_unique_key(rec):
    """Compose unique filesystem key. Three components separated by `__`:
      <tar_safe>__<key_in_tar_safe>__<phrase_hash16>

    Bijection: (dataset, tar, key_in_tar, phrase) → key is collision-free at ~9M scale
    (16 hex = 64-bit hash). The original (tar, key_in_tar, phrase) is ALSO stored as
    explicit fields in every manifest row, so reverse lookup never relies on the key alone.
    """
    tar = safe((rec.get('tar') or '').replace('.tar', ''))
    kit = safe(rec['key_in_tar'])
    ph16 = hashlib.md5(rec.get('phrase', '').encode()).hexdigest()[:16]
    return f"{tar}__{kit}__{ph16}" if tar else f"{kit}__{ph16}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', required=True, choices=list(DS_TO_SUB_SCHEMA.keys()))
    ap.add_argument('--out_dir', required=True, help='per-ds output dir; manifest+imgs go here')
    ap.add_argument('--shard', default='0/1', help='shard X/Y for sample-id-mod-Y distribution')
    ap.add_argument('--max_samples', type=int, default=None, help='cap per shard (debug)')
    args = ap.parse_args()

    sub, schema = DS_TO_SUB_SCHEMA[args.ds]
    tar_ds = DS_TO_TAR_DS[args.ds]
    si, sn = (int(x) for x in args.shard.split('/'))
    img_dir = os.path.join(args.out_dir, 'imgs')
    os.makedirs(img_dir, exist_ok=True)
    out_manifest = os.path.join(args.out_dir, f'manifest-shard-{si:03d}-of-{sn:03d}.jsonl')

    # load_rejected returns all F1<1 records randomized; shard by hash-mod-N for stability.
    load_ds = args.ds.replace('grit_v2', 'grit')  # load_rejected uses 'grit' for tar root mapping
    pool, n_files = load_rejected(load_ds, sub, schema, max_pool=10**9, seed=0)
    print(f'{args.ds}: pool={len(pool)} from {n_files} files; my shard {si}/{sn}', flush=True)

    n_done = n_skip = n_dup = 0
    seen = set()
    if os.path.isfile(out_manifest):
        for line in open(out_manifest):
            try: seen.add(json.loads(line)['key'])
            except: pass
        print(f'  resume: {len(seen)} already written', flush=True)

    mf = open(out_manifest, 'a')
    for i, rec in enumerate(pool):
        if hash(rec.get('key_in_tar', '') + rec.get('phrase', '')) % sn != si:
            continue
        if args.max_samples is not None and n_done >= args.max_samples: break
        uniq = make_unique_key(rec)
        if uniq in seen: n_dup += 1; continue
        try:
            tar_path = resolve_tar(tar_ds, rec['tar'])
            if not tar_path: n_skip += 1; continue
            img_bytes = extract_for_ds(tar_ds, tar_path, rec['key_in_tar'])
            if not img_bytes: n_skip += 1; continue
            kind = rec.get('kind', 'bbox')
            gt_b = rec['gt'] if kind == 'bbox' else None
            gt_p = rec['gt'] if kind == 'point' else None
            im = render_yellow_red(img_bytes, rec['pred'],
                                    gt_bboxes=gt_b, gt_points=gt_p,
                                    gt_norm=GT_NORM_BY_DS.get(args.ds, False),
                                    max_side=1024)
            out_jpg = os.path.join(img_dir, f'{args.ds}_{uniq}.jpg')
            im.save(out_jpg, 'JPEG', quality=85)
            mf.write(json.dumps({
                'dataset': args.ds, 'key': uniq,
                'tar': rec.get('tar', ''), 'key_in_tar': rec['key_in_tar'],
                'pair_idx': 0, 'phrase': rec['phrase'],
                'n_pred': len(rec['pred']), 'n_gt': len(rec['gt']),
                'f1': rec['f1'], 'precision': rec.get('precision', 0.0),
                'recall': rec.get('recall', 0.0), 'kind': rec['kind'],
                'gt': rec['gt'], 'pred': rec['pred'],
                'pred_raw': rec.get('pred_raw', ''),
                'img_path': out_jpg,
            }) + '\n')
            seen.add(uniq)
            n_done += 1
            if n_done % 500 == 0:
                mf.flush(); print(f'  [{args.ds} s{si}] {n_done} done · skip {n_skip} dup {n_dup}', flush=True)
        except Exception:
            n_skip += 1
    mf.close()
    # Post-write bijection assertion: every (tar, key_in_tar, phrase) appears exactly once
    keys_set = set(); composite_set = set()
    for line in open(out_manifest):
        r = json.loads(line)
        if r['key'] in keys_set:
            print(f'FATAL COLLISION on key={r["key"]}'); sys.exit(2)
        keys_set.add(r['key'])
        comp = (r['tar'], r['key_in_tar'], r['phrase'])
        if comp in composite_set:
            print(f'FATAL DUP composite (tar/key_in_tar/phrase)={comp}'); sys.exit(2)
        composite_set.add(comp)
    print(f'{args.ds} shard {si}/{sn}: rendered={n_done} skip={n_skip} dup={n_dup} '
          f'· bijection PASS ({len(keys_set)} unique) → {out_manifest}')


if __name__ == '__main__':
    main()
