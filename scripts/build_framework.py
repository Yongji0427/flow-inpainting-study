from pathlib import Path
import base64, io, html
import numpy as np
from PIL import Image

REPO=Path(__file__).resolve().parents[1]
OUT=REPO/'figures/main'
OUT.mkdir(parents=True, exist_ok=True)
tiles=np.load(REPO/'results/qualitative/test_qualitative_tiles.npz')['tiles']
S=[]
def add(s): S.append(s)
def rect(x,y,w,h,fill='#ffffff',stroke='#DCE2E8',r=12,sw=1.5):
 add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{r}" fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>')
def text(x,y,s,size=22,color='#263341',weight=400,anchor='start',family='Arial, Helvetica, sans-serif'):
 subs={'₀':'0','₁':'1','₂':'2','ₜ':'t','ₙ':'N'}
 rendered=''.join('<tspan baseline-shift="sub" font-size="70%">'+subs[c]+'</tspan>' if c in subs else html.escape(c) for c in s)
 rendered=rendered.replace('rθ','r<tspan baseline-shift="sub" font-size="70%">θ</tspan>')
 add(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" text-anchor="{anchor}" fill="{color}">{rendered}</text>')
def line(x1,y1,x2,y2,color='#8B98A5',width=2,dash='',arrow=False):
 add(f'<path d="M{x1},{y1} L{x2},{y2}" fill="none" stroke="{color}" stroke-width="{width}"'+(f' stroke-dasharray="{dash}"' if dash else '')+(f' marker-end="url(#arrow{color[1:]})"' if arrow else '')+'/>')
def path(d,color='#8B98A5',width=2,dash='',arrow=False):
 add(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"'+(f' stroke-dasharray="{dash}"' if dash else '')+(f' marker-end="url(#arrow{color[1:]})"' if arrow else '')+'/>')
def tile(row,col,x,y,size):
 a=np.rint(np.moveaxis((tiles[row,col]+1)/2,0,-1)*255).clip(0,255).astype('uint8')
 b=io.BytesIO();Image.fromarray(a).save(b,format='PNG')
 add(f'<image x="{x}" y="{y}" width="{size}" height="{size}" href="data:image/png;base64,{base64.b64encode(b.getvalue()).decode()}" style="image-rendering:pixelated"/>')
 rect(x,y,size,size,'none','#CCD4DC',0,1)
def pill(x,y,w,label,color,bg):
 rect(x,y,w,33,bg,'none',16);text(x+w/2,y+23,label,18,color,600,'middle')

add('<svg xmlns="http://www.w3.org/2000/svg" width="1600" height="1040" viewBox="0 0 1600 1040" role="img" aria-labelledby="title desc">')
add('<title id="title">One backbone, three restoration objectives</title><desc id="desc">Matched inputs and identical U-Net architectures; independently trained pixel regression, perceptual regression and conditional flow matching. Regression uses one forward pass. Flow matching integrates a masked velocity field with Euler sampling.</desc><defs>')
for c in ['8B98A5','4285F4','0F9D58','EA4335','263341']:
 add(f'<marker id="arrow{c}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse"><path d="M0 0 L10 5 L0 10 z" fill="#{c}"/></marker>')
add('</defs>')
rect(0,0,1600,1040,'white','none',0)
text(48,54,'One backbone, three restoration objectives',34,weight=700)
text(48,89,'Matched data, initial weights and update budgets · Each model is trained independently',21,'#657381')
text(1552,53,'METHOD OVERVIEW',16,'#657381',600,'end')
# Common input protocol
rect(48,122,540,306,'#FAFBFC')
text(72,160,'a',23,weight=700);text(101,160,'Matched observations',25,weight=600)
for x,label in [(76,'Original x'),(246,'Large hole'),(416,'Fragmented')]: text(x+62,199,label,19,'#536170',500,'middle')
tile(0,0,76,212,124);tile(0,1,246,212,124);tile(1,1,416,212,124)
line(205,275,239,275,arrow=True)
text(76,370,'y = (1 − m) ⊙ x',25,weight=500)
text(76,402,'64 × 64 RGB · 1,638 missing pixels per mask',19,'#657381')
line(589,275,628,275,arrow=True)
# Shared architecture
rect(640,122,912,306,'#FAFBFC')
text(664,160,'b',23,weight=700);text(695,160,'Common residual U-Net',25,weight=600)
pill(1324,138,202,'14.99M parameters','#536170','#EAF0F4')
text(662,210,'7-channel input',17,'#657381',500)
text(682,235,'[z, y, m]',21,weight=500)
line(687,275,758,275,arrow=True)
xs=[774,868,962,1056,1150,1244,1338];ys=[240,260,280,300,280,260,240];hs=[94,78,62,48,62,78,94];widths=[96,192,288,384,288,192,96]
for i in range(3):
 xx=xs[i]+27; xe=xs[6-i]+27; yy=205+i*27
 path(f'M{xx},{ys[i]-5} V{yy} H{xe} V{ys[6-i]-5}', '#A8B4BF',1.8,'5 5',True)
for i,(xx,yy,hh,n) in enumerate(zip(xs,ys,hs,widths)):
 rect(xx,yy,54,hh,'#DDE6EF' if i!=3 else '#AABDCF','#8DA1B4',5)
 text(xx+27,yy+hh+25,str(n),17,'#536170',500,'middle')
 if i<6:line(xx+57,yy+hh/2,xs[i+1]-6,ys[i+1]+hs[i+1]/2,arrow=True)
line(1397,287,1465,287,arrow=True)
text(1490,314,'RGB',18,'#657381',500,'middle')
rect(1470,250,40,44,'#EDF1F5','#8DA1B4',4)
text(1100,197,'Concatenated skip connections',17,'#657381',400,'middle')
text(675,405,'Time t → sinusoidal embedding → 128-D MLP',19,'#536170')
text(1518,405,'3 downsampling stages · 2 bottleneck blocks',17,'#657381',400,'end')
# Dispatch into independently trained objectives
line(1096,429,1096,462)
path('M292,493 V462 H1308 V493',arrow=False)
line(800,462,800,493,arrow=True)
line(292,479,292,496,arrow=True);line(1308,479,1308,496,arrow=True)

# Objective cards
cards=[(48,'c','Pixel regression','#4285F4','#F3F7FF'),(556,'d','Perceptual regression','#0F9D58','#F0F9F4'),(1064,'e','Conditional flow matching','#EA4335','#FFF5F3')]
for x,letter,label,c,bg in cards:
 rect(x,506,488,418,'white','#DCE2E8',12)
 path(f'M{x+14},507 H{x+474}',c,4)
 text(x+24,547,letter,22,c,700)
 text(x+56,547,label,25,'#263341',600)
 text(x+24,589,'TRAINING OBJECTIVE',16,'#77838E',600)
 if letter=='c':
  text(x+24,628,'Missing-region pixel MSE',25,c,600)
  text(x+24,663,'Predict clean pixels from visible context.',20,'#657381')
 elif letter=='d':
  text(x+24,628,'Pixel MSE + αc · feature loss',24,c,600)
  pill(x+24,643,225,'Frozen VGG16',c,bg)
  text(x+264,667,'Training only',18,'#657381')
 else:
  text(x+24,624,'Masked velocity MSE',25,c,600)
  text(x+24,651,'Target velocity: x − ε',20,'#536170')
  text(x+24,679,'zₜ = y + m ⊙ [(1 − t)ε + tx]',20,'#536170')
 line(x+24,693,x+464,693,'#E5E9EE',1)
 text(x+24,727,'INFERENCE',16,'#77838E',600)
 if letter in ['c','d']:
  pill(x+333,706,131,'NFE = 1',c,bg)
  text(x+46,779,'y, m',24,'#263341',500,'middle')
  line(x+82,770,x+137,770,c,2,arrow=True)
  rect(x+145,744,160,53,bg,c,7,1.5)
  text(x+225,778,'U-Net',24,c,600,'middle')
  line(x+313,770,x+368,770,c,2,arrow=True)
  text(x+409,779,'x̂',29,'#263341',500,'middle')
  text(x+24,842,'Input state z = y · time t = 0',22,'#536170')
  text(x+24,885,'x̂ = y + m ⊙ rθ(y, m)',25,'#263341')
 else:
  pill(x+270,706,194,'Variable NFE',c,bg)
  pts=[x+48,x+146,x+244,x+342,x+440]
  for j,xx in enumerate(pts):
   if j<4:line(xx+20,775,pts[j+1]-24,775,c,2,arrow=True)
   add(f'<circle cx="{xx}" cy="775" r="19" fill="{bg}" stroke="{c}" stroke-width="1.7"/>')
   text(xx,782,['z₀','z₁','z₂','…','zₙ'][j],18,c,500,'middle')
  text(x+24,830,'z₀ = y + m ⊙ ε',24,'#536170')
  text(x+24,866,'Masked Euler updates · Δt = 1/N',21,'#536170')
  text(x+24,901,'N ∈ {1, 2, 4, 8, 16, 32} · dev sweep',20,c,500)
# Footer invariant, evaluation
rect(48,950,1504,64,'#F4F6F8','none',10)
text(72,990,'Visible pixels preserved',22,'#263341',600)
text(367,990,'(1 − m) ⊙ x̂ = y',24,'#536170')
line(624,968,624,997,'#D5DDE5',1.5)
text(660,990,'Evaluate',18,'#657381',600)
text(767,990,'Masked PSNR ↑',22,'#263341',500)
text(1001,990,'Full-image LPIPS ↓',22,'#263341',500)
text(1285,990,'Latency & NFE',22,'#263341',500)
add('</svg>')
svg='\n'.join(S)
(OUT/'fig02_framework.svg').write_text(svg)
page='''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>OneFlow — Method overview</title><style>*{box-sizing:border-box}body{margin:0;background:#eef1f5;color:#263341;font-family:Arial,Helvetica,sans-serif}.toolbar{max-width:1664px;margin:0 auto;padding:24px 32px;display:flex;align-items:center;justify-content:space-between;gap:20px}.brand{font-size:15px;font-weight:700;letter-spacing:.06em}.links{display:flex;gap:10px}a{font-size:13px;text-decoration:none;color:#536170;border:1px solid #cdd5df;border-radius:6px;padding:9px 14px;background:white}main{max-width:1664px;padding:0 32px 32px;margin:auto}.sheet{background:white;border:1px solid #dce2e8;box-shadow:0 12px 48px #21344b0c;border-radius:10px;overflow:hidden}svg{display:block;width:100%;height:auto}.caption{max-width:1200px;margin:20px auto 0;line-height:1.6;font-size:14px;color:#657381}@media(max-width:800px){main{padding:0 12px 20px}.toolbar{padding:16px 12px}.sheet{overflow:auto}.sheet svg{min-width:1000px}}@media print{body{background:white}.toolbar,.caption{display:none}main{max-width:none;padding:0}.sheet{border:0;box-shadow:none;border-radius:0}svg{width:100%}@page{size:landscape;margin:6mm}}</style><body><header class="toolbar"><div class="brand">ONEFLOW / METHOD FIGURE</div><div class="links"><a href="fig02_framework.svg" download>Download SVG</a><a href="fig02_framework.png" download>Download PNG</a></div></header><main><div class="sheet">'''+svg+'''</div><p class="caption">Three independently trained models share the same residual U-Net architecture and matched training conditions. Pixel and perceptual regression use a single forward pass; conditional flow matching uses uniform Euler sampling. VGG16 supplies frozen training supervision only. Development evaluation selects NFE = 8 for the reported FM test comparison. Input thumbnails are archived observations of Abyssinian_2, with the original display mapping and no image retouching.</p></main></body></html>'''
(OUT/'fig02_framework.html').write_text(page)
print(OUT/'fig02_framework.html')
