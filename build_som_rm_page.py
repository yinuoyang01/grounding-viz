"""Standalone SoM RM data page: som-rm.html (user request 8/25).

Concatenates the SoM RM snippet panels (.snippets/cat3_<name>_{tab,panel}.html) into a
self-contained page, independent of generate.py/index.html. Viewed via
htmlpreview.github.io/?...som-rm.html.

    python3 build_som_rm_page.py
"""
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SNIP = os.path.join(HERE, '.snippets')
OUT = os.path.join(HERE, 'som-rm.html')
PANELS = ('som_v4', 'gui_verified', 'v44_cascade', 'v44_final')

CSS = '''
:root{--bg:#faf7f2;--fg:#222;--muted:#777;--line:#ddd;--accent2:#d6336c;}
body{font-family:-apple-system,'Segoe UI',Helvetica,Arial,sans-serif;background:var(--bg);
     color:var(--fg);margin:0;padding:24px 32px;}
h1{font-size:22px;} h2{font-size:17px;margin-top:28px;}
.ds-tabs{display:flex;gap:8px;flex-wrap:wrap;margin:14px 0 20px;}
.ds-tab{padding:7px 15px;border:1px solid var(--line);background:#fff;cursor:pointer;
        font-size:13px;border-radius:999px;color:var(--fg);}
.ds-tab:hover{border-color:var(--accent2);color:var(--accent2);}
.ds-tab.active{background:var(--accent2);color:#fff;border-color:var(--accent2);}
.panel{display:none;}
.panel.active{display:block;}
table{background:#fff;}
'''

JS = '''
document.querySelectorAll('.ds-tab').forEach(function(b){
  b.addEventListener('click', function(){
    document.querySelectorAll('.ds-tab').forEach(function(x){x.classList.remove('active');});
    document.querySelectorAll('.panel').forEach(function(p){p.classList.remove('active');});
    b.classList.add('active');
    var p = document.getElementById(b.dataset.panel);
    if (p) p.classList.add('active');
  });
});
var first = document.querySelector('.ds-tab');
if (first) first.click();
'''


def main():
    tabs, panels = [], []
    for name in PANELS:
        t = os.path.join(SNIP, f'cat3_{name}_tab.html')
        p = os.path.join(SNIP, f'cat3_{name}_panel.html')
        if os.path.isfile(t) and os.path.isfile(p):
            tabs.append(open(t).read())
            panels.append(open(p).read())
        else:
            print(f'  [skip] {name} (snippet missing)')
    html = ('<!DOCTYPE html><html><head><meta charset="utf-8">'
            '<title>SoM RM data</title>'
            f'<style>{CSS}</style></head><body>'
            '<h1 style="margin-top:0">SoM RM data</h1>'
            f'<div class="ds-tabs active">{"".join(tabs)}</div>'
            f'{"".join(panels)}'
            f'<script>{JS}</script></body></html>')
    open(OUT, 'w').write(html)
    print(f'wrote {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB, {len(tabs)} pills)')


if __name__ == '__main__':
    main()
