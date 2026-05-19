"""Build judge-pred-only manifest + render red-dot-only jpgs for ensemble VLM judge.

Samples N rejected (F1<1) records per dataset from grounding_filter_molmo2/{ds}_filtered/,
draws ONLY red pred dots (no GT — judge sees what Molmo saw), saves jpgs + manifest jsonl.

KEY DECISION: every record gets a FULL unique key (tar path + key_in_tar [+ phrase hash]).
Previously stripped to just sample_id, which caused collisions in:
  - grit_v2: same sample_id across 18% of coyo_X_snappy dirs (different images)
  - cc3m/cc12m: r0/r1/r2 regions per image (different phrases)
Result: img files were overwritten, judge saw phrase-A but red-dots-from-phrase-B.
"""
import argparse
import hashlib
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_filter import render_yellow_red, extract_image_from_tar, GT_NORM_BY_DS
from build_cat3_dataset_panels import resolve_tar, load_rejected, extract_for_ds

# (label, output_subdir, schema)
PLAN = [
    ('grit_v2',    'grit_v2_filtered',    'new'),
    ('cc3m',       'cc3m_filtered',       'new'),
    ('vg',         'vg_filtered',         'legacy'),
    ('openimages', 'openimages_filtered', 'legacy'),
    ('pixmo',      'pixmo_filtered',      'legacy'),
]


def make_unique_key(rec):
    """Compose tar + key_in_tar + phrase-hash so cross-tar / cross-region collisions disappear.
    phrase-hash needed for cc3m r0/r1 (both have same key_in_tar)."""
    tar = (rec.get('tar') or '').replace('/', '_').replace('.tar', '')
    key_in_tar = rec['key_in_tar']
    phrase_hash = hashlib.md5(rec.get('phrase', '').encode()).hexdigest()[:6]
    return f'{tar}__{key_in_tar}__{phrase_hash}' if tar else f'{key_in_tar}__{phrase_hash}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_grit_v2', type=int, default=1500)
    ap.add_argument('--n_cc3m',    type=int, default=1500)
    ap.add_argument('--n_vg',         type=int, default=700)
    ap.add_argument('--n_openimages', type=int, default=700)
    ap.add_argument('--n_pixmo',      type=int, default=600)
    ap.add_argument('--out_dir', default='/weka/oe-training-default/zixianm/yinuoy/grounding_rm/eval/results/red_5k_v3')
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()

    counts = {'grit_v2': args.n_grit_v2, 'cc3m': args.n_cc3m,
              'vg': args.n_vg, 'openimages': args.n_openimages, 'pixmo': args.n_pixmo}
    img_dir = os.path.join(args.out_dir, 'imgs')
    os.makedirs(img_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, 'manifest.jsonl')

    # Map for tar root lookup (grit_v2 shares grit tar root)
    tar_ds_map = {'grit_v2': 'grit', 'cc3m': 'cc3m', 'vg': 'vg', 'openimages': 'openimages', 'pixmo': 'pixmo'}

    total_done = total_skip = 0
    seen_keys = set()  # belt-and-suspenders dedup on the unique key
    with open(manifest_path, 'w') as mf:
        for ds, sub, schema in PLAN:
            n_want = counts[ds]
            # NB load_rejected expects rebuild_cat3 ds vocabulary (uses 'grit' not 'grit_v2')
            pool, n_files = load_rejected(ds.replace('grit_v2', 'grit'), sub, schema,
                                          max_pool=n_want * 4, seed=args.seed)
            print(f'{ds}: pool={len(pool)} from {n_files} files', flush=True)
            tar_ds = tar_ds_map[ds]
            n_done = n_skip = n_dup = 0
            for rec in pool:
                if n_done >= n_want: break
                uniq = make_unique_key(rec)
                if (ds, uniq) in seen_keys:
                    n_dup += 1; continue
                try:
                    tar_path = resolve_tar(tar_ds, rec['tar'])
                    if not tar_path: n_skip += 1; continue
                    img_bytes = extract_image_from_tar(tar_path, rec['key_in_tar'])
                    if not img_bytes: n_skip += 1; continue
                    # 2-stage judge needs YELLOW translucent GT + RED pred dots.
                    kind = rec.get('kind', 'bbox')
                    gt_bboxes = rec['gt'] if kind == 'bbox' else None
                    gt_points = rec['gt'] if kind == 'point' else None
                    im = render_yellow_red(img_bytes, rec['pred'],
                                            gt_bboxes=gt_bboxes, gt_points=gt_points,
                                            gt_norm=GT_NORM_BY_DS.get(ds, False),
                                            max_side=1024)
                    out_jpg = os.path.join(img_dir, f'{ds}_{uniq}.jpg')
                    im.save(out_jpg, 'JPEG', quality=85)
                    mf.write(json.dumps({
                        'dataset': ds,
                        'key': uniq,
                        'tar': rec.get('tar', ''),
                        'key_in_tar': rec['key_in_tar'],
                        'pair_idx': 0,
                        'phrase': rec['phrase'],
                        'n_pred': len(rec['pred']),
                        'n_gt': len(rec['gt']),
                        'f1': rec['f1'],
                        'precision': rec.get('precision', 0.0),
                        'recall': rec.get('recall', 0.0),
                        'kind': rec['kind'],
                        'gt': rec['gt'],         # carry GT coords for viz overlay
                        'pred': rec['pred'],     # carry pred coords for viz re-render
                        'pred_raw': rec.get('pred_raw', ''),
                        'img_path': out_jpg,
                    }) + '\n')
                    seen_keys.add((ds, uniq))
                    n_done += 1
                    if n_done % 200 == 0:
                        print(f'  [{ds}] {n_done}/{n_want} (skip {n_skip} dup {n_dup})', flush=True)
                except Exception as e:
                    n_skip += 1
            print(f'  {ds}: rendered={n_done}, skipped={n_skip}, dup={n_dup}', flush=True)
            total_done += n_done; total_skip += n_skip
    print(f'TOTAL: {total_done} rendered, {total_skip} skipped → {manifest_path}')


if __name__ == '__main__':
    main()
