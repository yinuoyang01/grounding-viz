"""14 pools x 10 sampled pairs from som_v47t_numbered_training -> v47t-pools.html."""
import base64, collections, io, json, random

from PIL import Image

D = '/weka/oe-training-default/zixianm/yinuoy/grounding_rm/data/som_v47t_numbered_training'
OUT = '/weka/oe-training-default/zixianm/yinuoy/grounding-viz/v47t-pools.html'
N = 10
MAXW = 640

by_pool = collections.defaultdict(list)
for l in open(f'{D}/train.jsonl'):
    r = json.loads(l)
    by_pool[r['meta']['pool']].append(r)

rng = random.Random(0)


def b64(path):
    im = Image.open(path).convert('RGB')
    if im.width > MAXW:
        im = im.resize((MAXW, int(im.height * MAXW / im.width)))
    buf = io.BytesIO()
    im.save(buf, format='JPEG', quality=78)
    return base64.b64encode(buf.getvalue()).decode()


tabs, panes = [], []
for pi, pool in enumerate(sorted(by_pool, key=lambda p: -len(by_pool[p]))):
    rows = rng.sample(by_pool[pool], min(N, len(by_pool[pool])))
    cards = []
    for r in rows:
        m = r['meta']
        rj_img, ch_img = r['images'][0], r['images'][1]
        stats = ' | '.join(f'{k}={m[k]}' for k in
                           ('f1_chosen', 'f1_rejected', 'ey_gap', 'tier', 'pts_chosen', 'pts_rejected')
                           if k in m)
        cards.append(f'''<div class="card">
<div class="hd"><b>{m.get("phrase","")}</b> <span class="dim">[{m.get("arena","?")}] {stats}</span></div>
<div class="pair">
<div><div class="lab good">CHOSEN</div><img src="data:image/jpeg;base64,{b64(ch_img)}"></div>
<div><div class="lab bad">REJECTED</div><img src="data:image/jpeg;base64,{b64(rj_img)}"></div>
</div></div>''')
    tabs.append(f'<button class="tab" onclick="show({pi})" id="t{pi}">{pool} ({len(by_pool[pool])})</button>')
    panes.append(f'<div class="pane" id="p{pi}">{"".join(cards)}</div>')

html = f'''<!doctype html><html><head><meta charset="utf-8"><title>v47g2 pools sample</title>
<style>
body{{font-family:sans-serif;margin:16px;background:#fafafa}}
.tab{{margin:2px;padding:6px 10px;border:1px solid #999;background:#eee;cursor:pointer;border-radius:4px}}
.tab.on{{background:#0a3235;color:#fff}}
.pane{{display:none}} .pane.on{{display:block}}
.card{{background:#fff;border:1px solid #ddd;border-radius:6px;margin:14px 0;padding:10px}}
.hd{{margin-bottom:8px}} .dim{{color:#777;font-size:13px}}
.pair{{display:flex;gap:12px;flex-wrap:wrap}}
.pair img{{max-width:640px;width:100%;border:1px solid #ccc}}
.pair>div{{flex:1;min-width:320px}}
.lab{{font-weight:bold;font-size:12px;margin-bottom:4px}}
.good{{color:#0a7a2f}} .bad{{color:#c22}}
</style></head><body>
<h2>som_v47t_numbered_training — {N} sampled pairs per pool (train 97,892)</h2>
<div>{''.join(tabs)}</div>{''.join(panes)}
<script>
function show(i){{document.querySelectorAll('.tab,.pane').forEach(e=>e.classList.remove('on'));
document.getElementById('t'+i).classList.add('on');document.getElementById('p'+i).classList.add('on');}}
show(0);
</script></body></html>'''
open(OUT, 'w').write(html)
print(OUT, len(html) / 1e6, 'MB', {p: len(v) for p, v in by_pool.items()})
