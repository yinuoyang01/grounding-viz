"""Render a cat3 viz tab for the 2-judge cascade (Qwen + InternVL) full-set output.

For each ds with cascade output in judge_full/<ds>/:
  - load judge_qwen_*.jsonl + judge_intern_*.jsonl, join on uniq_key
  - look up gt/pred from filter records (judge jsonl doesn't store coords)
  - render yellow GT + red pred image
  - categorize by 2-judge agreement:
      both_mask_wrong / both_yes / both_no / disagree
  - build cards

Output: .snippets/cat3_cascade_tab.html + cat3_cascade_panel.html
"""
import argparse
import base64
import glob
import hashlib
import html
import io
import json
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_filter import render_yellow_red, GT_NORM_BY_DS, GT_ORDER_BY_DS
from build_cat3_dataset_panels import resolve_tar, extract_for_ds, parse_new_key

JUDGE_ROOT = '/weka/oe-training-default/oe-encoder/judge_full'
FILTER_ROOT = '/weka/oe-training-default/oe-encoder/grounding_filter_molmo2'

# (ds, filter_subdir, filter_glob, schema, pretty)
DS_CFG = {
    'vg':         ('vg_filtered',         'worker_*.jsonl',          'legacy',   'Visual Genome'),
    'openimages': ('openimages_filtered', 'worker_*.jsonl',          'legacy',   'OpenImages V7'),
    'pixmo':      ('pixmo_filtered',      'worker_*.jsonl',          'legacy',   'Pixmo Point'),
    'grit_v2':    ('grit_v2_filtered',    '*.jsonl',                 'new',      'GRIT v2'),
    'cc3m':       ('cc3m_filtered_boost', '*.jsonl',                 'new',      'cc3m'),
    'cc12m':      ('cc12m_filtered',      '*.jsonl',                 'new',      'cc12m'),
    'seeclick':   ('seeclick_filtered',   '*.jsonl',                 'new',      'SeeClick'),
    'rf100':      ('rf100_filtered',      '*.jsonl',                 'new',      'RF100'),
}
DS_TO_TAR_DS = {'vg':'vg','openimages':'openimages','pixmo':'pixmo','grit_v2':'grit',
                'cc3m':'cc3m','cc12m':'cc12m','seeclick':'seeclick','rf100':'rf100'}
DS_ORDER = ['vg','openimages','pixmo','grit_v2','cc3m','cc12m','seeclick','rf100']


def uniq_key(tar, key_in_tar, phrase):
    ph = hashlib.md5((phrase or '').encode()).hexdigest()[:16]
    t = (tar or '').replace('/', '_').replace('.tar', '')
    k = (key_in_tar or '').replace('/', '_')
    return f'{t}__{k}__{ph}' if t else f'{k}__{ph}'


def load_filter_index(ds):
    """uniq_key → {tar, key_in_tar, gt, pred, kind, f1, pred_raw}."""
    sub, fglob, schema, _ = DS_CFG[ds]
    idx = {}
    for jf in sorted(glob.glob(f'{FILTER_ROOT}/{sub}/{fglob}')):
        for line in open(jf):
            try: r = json.loads(line)
            except: continue
            if r.get('error') or r.get('rescore_error'): continue
            if schema == 'rescored':
                tar, kit = r['shard'], r['key']
                gt = r.get('gt_bboxes') if r.get('kind') == 'bbox' else r.get('gt_points')
                pred = r.get('pred_pts') or []
                f1 = r.get('f1', 0.0)
            elif schema == 'legacy':
                tar, kit = r['shard'], r['key']
                gt = r.get('gt') or []
                pred = r.get('pred') or []
                f1 = (r.get('info') or {}).get('f1', 0.0)
            else:  # new
                tar, kit = parse_new_key(r['key'])
                if not tar: continue
                gt = r.get('gt_bboxes') if r.get('kind') == 'bbox' else r.get('gt_points')
                pred = r.get('pred_pts') or []
                f1 = r.get('f1', 0.0)
            uk = uniq_key(tar, kit, r.get('phrase', ''))
            idx[uk] = {'tar': tar, 'key_in_tar': kit, 'gt': gt or [], 'pred': pred,
                       'kind': r.get('kind', 'bbox'), 'f1': f1,
                       'pred_raw': r.get('pred_raw', '')}
    return idx


def load_judges(ds):
    """Return {uniq_key: {qwen: rec, intern: rec}}."""
    out = {}
    for mk in ('qwen', 'intern'):
        for jf in sorted(glob.glob(f'{JUDGE_ROOT}/{ds}/judge_{mk}_*.jsonl')):
            for line in open(jf):
                try: r = json.loads(line)
                except: continue
                if r.get('error'): continue
                uk = r['uniq_key']
                out.setdefault(uk, {})[mk] = r
    return out


def categorize(q, i):
    """2-judge category from qwen + intern molmo_verdict (mask is no longer judged)."""
    qv = q.get('molmo_verdict')
    iv = i.get('molmo_verdict')
    if qv == 'yes' and iv == 'yes': return 'both_yes'
    if qv == 'no' and iv == 'no': return 'both_no'
    return 'disagree'


def render_card(ds, fr, q, i):
    tar_ds = DS_TO_TAR_DS.get(ds, ds)
    tar_path = resolve_tar(tar_ds, fr['tar'])
    if not tar_path: return None
    img_bytes = extract_for_ds(tar_ds, tar_path, fr['key_in_tar'])
    if not img_bytes: return None
    kind = fr['kind']
    gt_b = fr['gt'] if kind == 'bbox' else None
    gt_p = fr['gt'] if kind == 'point' else None
    try:
        im = render_yellow_red(img_bytes, fr['pred'], gt_bboxes=gt_b, gt_points=gt_p,
                               gt_norm=GT_NORM_BY_DS.get(ds, False),
                               gt_order=GT_ORDER_BY_DS.get(ds, 'xy'), max_side=400)
    except Exception:
        return None
    buf = io.BytesIO(); im.save(buf, 'JPEG', quality=80)
    b64 = base64.b64encode(buf.getvalue()).decode()
    # clean original (no marks) so the viewer can see what the marks occlude
    try:
        from PIL import Image as _Im
        clean = _Im.open(io.BytesIO(img_bytes)).convert('RGB')
        if max(clean.size) > 400:
            _s = 400 / max(clean.size)
            clean = clean.resize((int(clean.size[0]*_s), int(clean.size[1]*_s)))
        _cb = io.BytesIO(); clean.save(_cb, 'JPEG', quality=80)
        b64_clean = base64.b64encode(_cb.getvalue()).decode()
    except Exception:
        b64_clean = b64

    def chip(label, v):
        bg = {'yes':'#0FCB8C','no':'#F0529C','unclear':'#B11BE8',
              'mask_ok':'#0FCB8C','mask_wrong':'#F0529C','mask_unclear':'#B11BE8'}.get(v,'#999')
        return (f'<span style="background:{bg};color:#FAF2E9;padding:2px 7px;border-radius:4px;'
                f'font-size:10px;font-weight:700;margin-right:4px">{label}: {html.escape(str(v))}</span>')

    def jrow(name, jr):
        return (f'<div style="margin-bottom:3px"><span style="display:inline-block;width:70px;'
                f'font-size:11px;font-weight:600;color:#0A3235">{name}:</span>'
                f'{chip("molmo", jr.get("molmo_verdict","?"))}</div>')

    reasons = ''
    for nm, jr in [('Qwen', q), ('InternVL', i)]:
        rs = jr.get('reason', '')
        if rs:
            reasons += f'<div style="font-size:10px;color:rgba(10,50,53,0.65);margin-top:2px"><b>{nm}:</b> {html.escape(rs)}</div>'

    ST = ('display:grid;grid-template-columns:720px 1fr;gap:16px;padding:12px;margin:8px 0;'
          'background:var(--bg,#FAF2E9);border:1px solid rgba(10,50,53,0.15);border-radius:8px')
    phrase = html.escape((q.get('phrase') or '')[:300])
    return (f'<div style="{ST}">'
            f'<div style="display:flex;gap:8px">'
            f'<img src="data:image/jpeg;base64,{b64_clean}" style="width:50%;border-radius:4px"/>'
            f'<img src="data:image/jpeg;base64,{b64}" style="width:50%;border-radius:4px"/>'
            f'</div>'
            '<div>'
            f'<div style="color:rgba(10,50,53,0.55);font-family:ui-monospace,Menlo,monospace;font-size:11px;margin-bottom:6px">{ds} / {html.escape(fr["key_in_tar"])} · F1={fr["f1"]:.2f}</div>'
            f'<div style="color:#0A3235;font-size:13px;margin-bottom:6px"><b>{phrase}</b></div>'
            f'{jrow("Qwen", q)}{jrow("InternVL", i)}'
            f'<div style="margin-top:4px">{reasons}</div>'
            '</div></div>')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ds', nargs='+', default=DS_ORDER, help='datasets to render (default: all 8)')
    ap.add_argument('--per_cat', type=int, default=12)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--snippets_dir', default='/weka/oe-training-default/zixianm/yinuoy/grounding-viz/.snippets')
    args = ap.parse_args()
    rng = random.Random(args.seed)

    from collections import defaultdict
    panels = []
    tabs = ''
    order = [('both_yes', '✅ Both molmo=yes (false-reject — Molmo correct)'),
             ('both_no', '❌ Both molmo=no (real hard example — Molmo wrong)'),
             ('disagree', '❓ Qwen↔InternVL disagree (→ GPT-5 arbitrate)')]

    for ds in args.ds:
        pretty = DS_CFG[ds][3]
        tabs += f'<button class="ds-tab" data-panel="p_3_{ds}">{html.escape(pretty)}</button>'
        judges = load_judges(ds)
        both = {uk: v for uk, v in judges.items() if 'qwen' in v and 'intern' in v}
        print(f'{ds}: {len(judges)} judged, {len(both)} by both qwen+intern', flush=True)

        if not both:
            # placeholder: cascade hasn't produced 2-judge output for this ds yet
            panels.append(
                f'<div id="p_3_{ds}" class="panel" data-cat="cat3">'
                f'<div class="dataset-intro"><div class="intro-title"><b>{pretty}</b></div>'
                f'<div class="intro-desc">2-judge cascade (Qwen3-VL-32B + InternVL3-78B) on F1&lt;1 pool — '
                f'not yet judged (in queue / Tier-2 awaiting).</div></div></div>\n')
            continue

        filt = load_filter_index(ds)
        print(f'  filter index: {len(filt)}', flush=True)
        cats = defaultdict(list)
        for uk, v in both.items():
            cats[categorize(v['qwen'], v['intern'])].append(uk)

        # 3 sub-tabs (one per category) inside this dataset panel
        SUBLABEL = {
            'both_yes':  'Both molmo=yes',
            'both_no':   'Both molmo=no',
            'disagree':  'Judges disagree',
        }
        sub_btns, sub_panels = [], []
        for si, (cat, _full) in enumerate(order):
            uks = cats.get(cat, [])
            picks = rng.sample(uks, min(args.per_cat, len(uks))) if uks else []
            cards = []
            for uk in picks:
                fr = filt.get(uk)
                if not fr: continue
                c = render_card(ds, fr, both[uk]['qwen'], both[uk]['intern'])
                if c: cards.append(c)
            spid = f'sub_{ds}_{cat}'
            active = ' active' if si == 0 else ''
            disp = '' if si == 0 else 'display:none'
            sub_btns.append(
                f'<button class="sub-tab{active}" onclick="showCascadeSub(\'{ds}\',\'{cat}\',this)">'
                f'{SUBLABEL[cat]} ({len(uks)})</button>')
            body = "".join(cards) if cards else '<div style="padding:16px;color:rgba(10,50,53,0.5)">No samples in this category yet.</div>'
            sub_panels.append(
                f'<div id="{spid}" class="cascade-sub" data-ds="{ds}" style="{disp}">'
                f'<div style="font-size:11px;color:rgba(10,50,53,0.55);margin:6px 0">'
                f'{len(uks):,} total · showing {len(cards)}</div>{body}</div>')

        n_total = len(both)
        intro = ('<div class="dataset-intro">'
                 f'<div class="intro-title"><b>{pretty} — F1&lt;1 pool, 2-judge cascade (Qwen3-VL-32B + InternVL3-78B)</b></div>'
                 f'<div class="intro-desc">Card shows yellow GT + red Molmo pred for reference, but the judges '
                 f'saw ONLY the red pred (no GT mask) — molmo_verdict + reason. 3 sub-tabs by agreement.</div>'
                 f'<div class="intro-meta">{n_total:,} F1&lt;1 samples judged by both · '
                 + ' · '.join(f'{SUBLABEL[c]}={len(cats.get(c,[]))}' for c,_ in order) + '</div></div>')
        panels.append(
            f'<div id="p_3_{ds}" class="panel" data-cat="cat3">\n{intro}\n'
            f'<div class="sub-tabs">{"".join(sub_btns)}</div>\n'
            f'{"".join(sub_panels)}\n</div>\n')

    # Inline JS for sub-tab switching (self-contained — no dependency on generate.py JS)
    sub_js = """<script>
function showCascadeSub(ds, cat, btn) {
  document.querySelectorAll('.cascade-sub[data-ds="' + ds + '"]').forEach(function(p){ p.style.display='none'; });
  var el = document.getElementById('sub_' + ds + '_' + cat); if (el) el.style.display='';
  var tabs = btn.parentElement.querySelectorAll('.sub-tab');
  tabs.forEach(function(t){ t.classList.remove('active'); });
  btn.classList.add('active');
}
</script>
"""
    os.makedirs(args.snippets_dir, exist_ok=True)
    # Overwrite the cat3 dataset snippets (generate.py injects these)
    with open(os.path.join(args.snippets_dir, 'cat3_dataset_tabs.html'), 'w') as f:
        f.write(f'<div id="cat3_tabs" class="ds-tabs">{tabs}</div>\n')
    with open(os.path.join(args.snippets_dir, 'cat3_dataset_panels.html'), 'w') as f:
        f.write(sub_js + '\n'.join(panels))
    print(f'wrote cat3_dataset_tabs.html + cat3_dataset_panels.html '
          f'({sum(len(p) for p in panels):,}b, {len(panels)} panels)')


if __name__ == '__main__':
    main()
