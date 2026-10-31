#!/usr/bin/env python3
"""Redraw the teaser from archived development scores; no model evaluation."""
from pathlib import Path
import csv
import hashlib
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'
plt.rcParams.update({'pdf.fonttype':42, 'font.size':8, 'axes.titlesize':8,
 'axes.labelsize':8, 'xtick.labelsize':7, 'ytick.labelsize':7,
 'axes.spines.top':True, 'axes.spines.right':True, 'axes.linewidth':.65,
 'axes.edgecolor':'#555555', 'xtick.color':'#444444','ytick.color':'#444444',
 'legend.frameon':False, 'savefig.facecolor':'white'})
REPO=Path(__file__).resolve().parents[1]
DATA=REPO/'results/development'
VALUES=REPO/'results/figure_values'
VALUES.mkdir(parents=True,exist_ok=True)
ROOT=REPO/'figures/main'
ROOT.mkdir(parents=True,exist_ok=True)
with (DATA/'figures_nfe_source.csv').open() as f:
    rows=[r for r in csv.DictReader(f) if r['model']=='fm']
NFE=np.array([1,2,4,8,16,32])
means={}
for metric in ['psnr_db','lpips']:
    curves=[]
    for seed in ['0','1']:
        for geometry in ['large','fragmented']:
            subset={int(r['nfe']):r for r in rows if r['seed']==seed and r['geometry']==geometry}
            assert sorted(subset)==NFE.tolist()
            assert all(int(r['dev_images'])==124 and int(r['mask_pairs_per_image'])==4 and int(r['noise_samples'])==4 for r in subset.values())
            curves.append([float(subset[n][metric]) for n in NFE])
    means[metric]=np.array(curves).mean(axis=0)
psnr=means['psnr_db']; lpips=means['lpips']
assert NFE[psnr.argmax()]==2 and NFE[lpips.argmin()]==8
norm_psnr=(psnr-psnr.min())/(psnr.max()-psnr.min())
norm_lpips=(lpips.max()-lpips)/(lpips.max()-lpips.min())
with (VALUES/'fig01b_plotted_values.csv').open('w',newline='') as f:
    w=csv.writer(f); w.writerow(['nfe','mean_psnr_db','mean_lpips','normalized_psnr','normalized_reversed_lpips'])
    w.writerows(zip(NFE,psnr,lpips,norm_psnr,norm_lpips))
PSNR='#4285F4'; LPIPS='#EA4335'; GRID='#E8E8E8'
def base(ax):
    ax.set_xscale('log',base=2)
    ax.set_xticks(NFE,[str(n) for n in NFE])
    ax.set_xlim(.9,36)
    ax.tick_params(length=3,width=.6,pad=3,direction="in")
    ax.grid(axis='y',color=GRID,lw=.55)
    ax.set_axisbelow(True)
def draw(ax,values,color,marker):
    ax.plot(NFE,values,color=color,lw=1.45,marker=marker,ms=3,mew=.6,
            label="LPIPS" if color==LPIPS else "PSNR")
def highlight(ax,n,y,color,marker):
    ax.plot(n,y,marker=marker,ms=4.4,mfc='white',mec=color,mew=.9,zorder=5)
def save(fig,name):
    fig.canvas.draw()
    for ext in ['pdf','svg','png']:
        fig.savefig(ROOT/f'{name}.{ext}',dpi=400)
    plt.close(fig)
# A: compact reconstruction retaining the existing common normalized axis.
fig,ax=plt.subplots(figsize=(3.25,2.30))
fig.subplots_adjust(left=.17,right=.97,bottom=.21,top=.96)
base(ax)
draw(ax,norm_lpips,LPIPS,'o'); draw(ax,norm_psnr,PSNR,'s')
highlight(ax,2,1,PSNR,'s'); highlight(ax,8,1,LPIPS,'o')
ax.set_ylim(-.045,1.14); ax.set_yticks([0,.5,1],['0','0.5','1'])
ax.set_ylabel('Normalized quality ↑',labelpad=4)
ax.set_xlabel('Function evaluations (NFE)',labelpad=4)
# Keep the original centered label geometry; color only the budget number.
from matplotlib.font_manager import FontProperties
def peak_label(ax, n, metric, color):
    prefix = f'Best {metric}: '
    number = str(n)
    renderer = ax.figure.canvas.get_renderer()
    font = FontProperties(family='sans-serif', size=6.7)
    def width(text):
        return renderer.get_text_width_height_descent(text, font, False)[0] * 72 / ax.figure.dpi
    left = -width(prefix + number) / 2
    for text, offset, text_color in [(prefix, left, '#000000'),
                                     (number, left + width(prefix), color)]:
        ax.annotate(text, xy=(n,1), xytext=(offset,7), textcoords='offset points',
                    ha='left', fontsize=6.7, color=text_color)
peak_label(ax, 2, 'PSNR', PSNR)
peak_label(ax, 8, 'LPIPS', LPIPS)
ax.legend(loc='lower center',bbox_to_anchor=(.46,.10),fontsize=7,
          handlelength=1.7,labelspacing=.4,borderpad=0)
save(fig,'fig01b_sampling_budget')
meta={
 'source_file':'results/development/figures_nfe_source.csv', 'source_sha256':hashlib.sha256((DATA/'figures_nfe_source.csv').read_bytes()).hexdigest(),
 'original_source':'results/development/figures_nfe_source.csv',
 'split':'development', 'images':124, 'training_seeds':[0,1], 'geometries':['large','fragmented'],
 'mask_pairs_per_image':4,'noise_samples_per_condition':4,
 'aggregation':'Arithmetic mean of the four archived seed-by-geometry means at each NFE. Scores, not generated images, were averaged upstream.',
 'normalization':'After seed/geometry averaging, PSNR=(x-min)/(max-min); LPIPS=(max-x)/(max-min), over the six plotted NFE settings.',
 'uncertainty':'Only means shown. No error bands or confidence intervals added.',
 'outputs':'fig01b_sampling_budget shows normalized mean quality. PDF/SVG contain vector lines and text.',
 'new_inference':False,'manuscript_replaced':True
}
(VALUES/'fig01b_provenance.json').write_text(json.dumps(meta,indent=2)+'\n')
print('Rendered Figure 1b from archived values; no inference.')
