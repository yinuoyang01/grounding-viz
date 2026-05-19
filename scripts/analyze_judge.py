"""Analyze 4-judge ensemble v3 results (2-stage: mask_verdict + molmo_verdict).

Tables:
  1. Per-judge verdict breakdown (mask_ok rate, molmo yes/no/unclear among mask_ok)
  2. Pairwise agreement on mask_verdict and molmo_verdict
  3. 4-judge majority categorization:
       all_mask_wrong  → drop, GT broken
       maj_mask_wrong  → likely drop
       all_yes / maj_yes (mask_ok + molmo_yes) → SAFE RECOVER (Molmo correct on real GT)
       all_no / maj_no  → HARD EXAMPLE (Molmo wrong on real GT)
       tied / unclear   → ambiguous
  4. Per-dataset breakdown of above

Outputs a markdown report to stdout + (optional) json summary.
"""
import argparse
import json
import os
from collections import defaultdict, Counter

OUT = '/weka/oe-training-default/zixianm/yinuoy/grounding_rm/eval/results/red_5k_v3'

JUDGES = [
    ('gpt5',    'judge_gpt5.jsonl',                 'GPT-5'),
    ('qwen',    'judge_qwen3vl_32b_instruct.jsonl', 'Qwen3-VL-32B'),
    ('intern',  'judge_internvl3_78b.jsonl',        'InternVL3-78B'),
    ('glm',     'judge_glm4v_108b.jsonl',           'GLM-4.5V'),
]


def load(name):
    p = os.path.join(OUT, name)
    if not os.path.isfile(p): return {}
    d = {}
    for line in open(p):
        try: r = json.loads(line)
        except: continue
        k = (r['dataset'], str(r['key']), int(r.get('pair_idx', 0)))
        d[k] = r
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', default=os.path.join(OUT, 'manifest.jsonl'))
    ap.add_argument('--save_json', default=None)
    args = ap.parse_args()

    manifest = {}
    for line in open(args.manifest):
        r = json.loads(line)
        manifest[(r['dataset'], str(r['key']), int(r.get('pair_idx', 0)))] = r
    print(f'manifest: {len(manifest)} rows')

    judges = {sn: load(fn) for sn, fn, _ in JUDGES}
    for sn, _, pretty in JUDGES:
        print(f'  {pretty}: {len(judges[sn])}/{len(manifest)} done')

    base = set(manifest)
    for sn, _, _ in JUDGES:
        base &= set(judges[sn])
    print(f'\n4-judge intersection: {len(base)}')
    if not base:
        print('No samples judged by all 4 yet; pick what is available'); return

    # ============================ Per-judge breakdown ============================
    print('\n## Per-judge verdict breakdown')
    print(f"{'judge':18s} {'mask_ok':>10s} {'mask_wrong':>12s} {'mol_yes':>10s} {'mol_no':>10s} {'mol_unclear':>14s}")
    for sn, _, pretty in JUDGES:
        jd = judges[sn]
        mv = Counter(jd[k].get('mask_verdict', 'mask_unclear') for k in base)
        mov = Counter(jd[k].get('molmo_verdict', 'unclear') for k in base)
        n = len(base)
        print(f"{pretty:18s} "
              f"{mv['mask_ok']:>5d} ({100*mv['mask_ok']/n:>4.1f}%)  "
              f"{mv['mask_wrong']:>5d} ({100*mv['mask_wrong']/n:>4.1f}%)  "
              f"{mov['yes']:>5d} ({100*mov['yes']/n:>4.1f}%)  "
              f"{mov['no']:>5d} ({100*mov['no']/n:>4.1f}%)  "
              f"{mov['unclear']:>5d} ({100*mov['unclear']/n:>4.1f}%)")

    # ============================ Pairwise mask_verdict agreement ============================
    print('\n## Pairwise mask_verdict agreement')
    print(f"{'':18s}", *[f"{p[:8]:>10s}" for _, _, p in JUDGES])
    for sa, _, pa in JUDGES:
        row = [f"{pa[:18]:18s}"]
        for sb, _, _ in JUDGES:
            if sa == sb: row.append('       --'); continue
            agree = sum(1 for k in base if judges[sa][k].get('mask_verdict') == judges[sb][k].get('mask_verdict'))
            row.append(f"{100*agree/len(base):>9.1f}%")
        print(' '.join(row))

    print('\n## Pairwise molmo_verdict agreement (only where BOTH said mask_ok)')
    for sa, _, pa in JUDGES:
        row = [f"{pa[:18]:18s}"]
        for sb, _, _ in JUDGES:
            if sa == sb: row.append('       --'); continue
            mol_ok = [k for k in base
                      if judges[sa][k].get('mask_verdict') == 'mask_ok'
                      and judges[sb][k].get('mask_verdict') == 'mask_ok']
            if not mol_ok: row.append('       n/a'); continue
            agree = sum(1 for k in mol_ok if judges[sa][k].get('molmo_verdict') == judges[sb][k].get('molmo_verdict'))
            row.append(f"{100*agree/len(mol_ok):>9.1f}%")
        print(' '.join(row))

    # ============================ Ensemble categorization ============================
    cats = defaultdict(list)
    for k in base:
        mvs = [judges[sn][k].get('mask_verdict', 'mask_unclear') for sn, _, _ in JUDGES]
        movs = [judges[sn][k].get('molmo_verdict', 'unclear') for sn, _, _ in JUDGES]
        n_mask_ok = sum(v == 'mask_ok' for v in mvs)
        n_mask_wrong = sum(v == 'mask_wrong' for v in mvs)
        n_yes = sum(v == 'yes' for v in movs)
        n_no = sum(v == 'no' for v in movs)
        if n_mask_wrong == 4: cat = 'all_mask_wrong'
        elif n_mask_wrong >= 3: cat = 'maj_mask_wrong'
        elif n_yes == 4: cat = 'all_yes'
        elif n_no == 4: cat = 'all_no'
        elif n_yes >= 3: cat = 'maj_yes'
        elif n_no >= 3: cat = 'maj_no'
        else: cat = 'tied'
        cats[cat].append(k)

    n = len(base)
    print(f'\n## 4-judge ensemble category (n={n})')
    order = ['all_mask_wrong', 'maj_mask_wrong', 'all_yes', 'maj_yes', 'tied', 'maj_no', 'all_no']
    labels = {
        'all_mask_wrong': '🚫 ALL mask_wrong (DROP — GT broken consensus)',
        'maj_mask_wrong': '⚠️  Maj mask_wrong (likely drop)',
        'all_yes':        '✅ ALL molmo=yes (RECOVER — Molmo correct, GT was the problem)',
        'maj_yes':        '✅ Maj molmo=yes (likely recover)',
        'tied':           '❓ Tied / ambiguous',
        'maj_no':         '❌ Maj molmo=no (likely hard example)',
        'all_no':         '❌ ALL molmo=no (HARD EXAMPLE — Molmo wrong on real GT)',
    }
    for c in order:
        cnt = len(cats[c])
        print(f"  {labels[c]:60s} {cnt:5d} ({100*cnt/n:.1f}%)")

    # ============================ Per-dataset breakdown ============================
    print('\n## Per-dataset breakdown')
    by_ds = defaultdict(lambda: Counter())
    for c, ks in cats.items():
        for k in ks: by_ds[manifest[k]['dataset']][c] += 1
    print(f"  {'ds':12s} {'n':>5s} {'%mask_wr':>10s} {'%all_yes':>10s} {'%maj_yes':>10s} {'%tied':>8s} {'%maj_no':>9s} {'%all_no':>9s}")
    for ds in sorted(by_ds):
        cc = by_ds[ds]; t = sum(cc.values())
        mw = (cc['all_mask_wrong'] + cc['maj_mask_wrong'])
        print(f"  {ds:12s} {t:5d} "
              f"{100*mw/t:>8.1f}%  "
              f"{100*cc['all_yes']/t:>8.1f}%  "
              f"{100*cc['maj_yes']/t:>8.1f}%  "
              f"{100*cc['tied']/t:>7.1f}% "
              f"{100*cc['maj_no']/t:>7.1f}% "
              f"{100*cc['all_no']/t:>7.1f}%")

    if args.save_json:
        out = {'n_intersection': n, 'cats': {c: len(ks) for c, ks in cats.items()},
               'per_dataset': {ds: dict(cc) for ds, cc in by_ds.items()}}
        with open(args.save_json, 'w') as f: json.dump(out, f, indent=2)
        print(f'\nSaved summary → {args.save_json}')


if __name__ == '__main__':
    main()
