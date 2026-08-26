"""constructs-v47.html: human review page for the v47 construction layers.

Eight tabs (4 chrisc real-mask kinds + 4 GUISyn kinds), N pairs each:
instruction, chosen render, rejected render side by side. Images downscaled and
inlined as base64 so the page is self-contained for htmlpreview.
"""
import base64
import io
import json
import random

from PIL import Image

Image.MAX_IMAGE_PIXELS = None
G = '/weka/oe-training-default/zixianm/yinuoy/grounding_rm'
SOURCES = [
    ('chrisc 真 mask', f'{G}/data/masked_pairs_v47/pairs.jsonl',
     ['cross_label', 'displaced', 'over_marked', 'absent'], f'{G}/data'),
    ('GUISyn 构造', '/tmp/claude-0/-weka-oe-training-default-zixianm-yinuoy/c1690b99-8811-491b-9a84-0a9a1a0aff41/scratchpad/gui_smoke/pairs.jsonl',
     ['wrongel', 'nearmiss', 'multi', 'abstain_nt'], None),
]
N_PER = 8
MAXW = 640

CSS = '''
:root{--bg:#faf7f2;--fg:#222;--muted:#777;--line:#ddd;--accent2:#d6336c;}
body{font-family:-apple-system,'Segoe UI',Helvetica,Arial,sans-serif;background:var(--bg);
     color:var(--fg);margin:0;padding:24px 32px;}
h1{font-size:22px;} .note{color:var(--muted);font-size:13px;margin-bottom:14px;}
.ds-tabs{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0 20px;}
.ds-tab{padding:7px 15px;border:1px solid var(--line);background:#fff;cursor:pointer;
        font-size:13px;border-radius:999px;}
.ds-tab.active{background:var(--accent2);color:#fff;border-color:var(--accent2);}
.panel{display:none;} .panel.active{display:block;}
.pair{background:#fff;border:1px solid var(--line);border-radius:8px;padding:14px;margin-bottom:18px;}
.pair h3{font-size:14px;margin:0 0 8px;}
.duo{display:flex;gap:12px;flex-wrap:wrap;}
.duo figure{margin:0;flex:1;min-width:280px;}
.duo img{max-width:100%;border:1px solid var(--line);}
figcaption{font-size:12px;margin-bottom:4px;}
.good{color:#2b8a3e;font-weight:600;} .bad{color:#c92a2a;font-weight:600;}
'''
JS = '''
document.querySelectorAll('.ds-tab').forEach(function(b){
  b.addEventListener('click', function(){
    document.querySelectorAll('.ds-tab').forEach(function(x){x.classList.remove('active');});
    document.querySelectorAll('.panel').forEach(function(p){p.classList.remove('active');});
    b.classList.add('active');
    document.getElementById(b.dataset.panel).classList.add('active');
  });
});
var f=document.querySelector('.ds-tab'); if(f) f.click();
'''

DESC = {
    'cross_label': 'rejected = 同图另一组人工标注物体的点(等数量);真 mask 验证不落在目标上',
    'displaced': 'rejected = 每个点推出它自己的人工 mask、落在物体外',
    'over_marked': 'rejected = 全对的点 + 借来的错误点(验证不在目标 mask 上)',
    'absent': 'instruction 是图里不存在的负短语;chosen 零标记,rejected 有标记',
    'wrongel': 'rejected = 同截图另一个具名控件的中心(离目标框 > ring 半径)',
    'nearmiss': 'rejected = 紧贴目标框外的点(ring 半径+2 ~ 25% 对角线)',
    'multi': '同名控件 k>=2:chosen 全标对,rejected 换错其中一个(截图上多标记也能赢)',
    'abstain_nt': 'instruction 借自其他截图(验证本图无此控件);chosen 零标记',
}


def b64(path, base):
    p = path if path.startswith('/') else f'{base}/{path}'
    im = Image.open(p).convert('RGB')
    if im.width > MAXW:
        im = im.resize((MAXW, round(im.height * MAXW / im.width)))
    buf = io.BytesIO()
    im.save(buf, 'JPEG', quality=78)
    return base64.b64encode(buf.getvalue()).decode()


def main():
    rng = random.Random(11)
    tabs, panels = [], []
    for group, path, kinds, base in SOURCES:
        by = {k: [] for k in kinds}
        for l in open(path):
            r = json.loads(l)
            k = r['meta']['kind']
            if k in by:
                by[k].append(r)
        for k in kinds:
            rows = by[k]
            rng.shuffle(rows)
            pid = f'p_{k}'
            tabs.append(f'<button class="ds-tab" data-panel="{pid}">{k} ({len(rows)})</button>')
            cards = [f'<p class="note">{group} · {DESC[k]} · 池 {len(rows)} 对,抽 {min(N_PER,len(rows))}</p>']
            for r in rows[:N_PER]:
                m = r['meta']
                ch = b64(r['images'][1], base or '')
                rj = b64(r['images'][0], base or '')
                cards.append(f'''<div class="pair"><h3>"{m['phrase']}"</h3><div class="duo">
<figure><figcaption class="good">chosen ({m['pts_chosen']} marks)</figcaption>
<img src="data:image/jpeg;base64,{ch}"></figure>
<figure><figcaption class="bad">rejected ({m['pts_rejected']} marks)</figcaption>
<img src="data:image/jpeg;base64,{rj}"></figure></div></div>''')
            panels.append(f'<div class="panel" id="{pid}">' + '\n'.join(cards) + '</div>')
    html = f'''<!doctype html><html><head><meta charset="utf-8">
<title>v47 construction layers</title><style>{CSS}</style></head><body>
<h1>v47 构造层人审 · 8 类 × {N_PER} 对</h1>
<p class="note">左 chosen / 右 rejected · som_marks_v3 (ring α115) · 全部几何/真mask判定,零判官</p>
<div class="ds-tabs">{''.join(tabs)}</div>
{''.join(panels)}
<script>{JS}</script></body></html>'''
    open('/weka/oe-training-default/zixianm/yinuoy/grounding-viz/constructs-v47.html', 'w').write(html)
    print('wrote constructs-v47.html', len(html) // 1024, 'KB')


if __name__ == '__main__':
    main()
