"""Render Figure 5 from verified, archived predictions; no model inference.
Run: python scripts/plot_qualitative_results.py
Dependencies: numpy, matplotlib.
"""
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
root=Path(__file__).resolve().parents[1]
(root/'figures/main').mkdir(parents=True,exist_ok=True)
tiles=np.load(root/'results/qualitative/test_qualitative_tiles.npz')['tiles']
assert tiles.shape==(4,5,3,64,64)
plt.rcParams.update({'font.family':'sans-serif','font.sans-serif':['Arial','DejaVu Sans'],
                     'font.size':8,'pdf.fonttype':42,'svg.fonttype':'none','savefig.facecolor':'white'})
fig,axes=plt.subplots(4,5,figsize=(6.5,4.65))
fig.subplots_adjust(left=.125,right=.997,bottom=.015,top=.935,wspace=.04,hspace=.12)
columns=['Ground truth','Masked input','MSE Regression','Perceptual Regression','FM8']
colors=['#262626','#262626','#4285F4','#0F9D58','#EA4335']
labels=['Favorable\nLarge hole','Favorable\nFragmented','Failure\nLarge hole','Failure\nFragmented']
for r in range(4):
 for c in range(5):
  ax=axes[r,c]
  ax.imshow(np.moveaxis((tiles[r,c]+1)/2,0,-1),interpolation='nearest')
  ax.set_axis_off()
  if r==0:
   ax.set_title(columns[c],fontsize=7.6,pad=8,color='#262626')
   if c>=2:ax.plot([0,1],[1.035,1.035],transform=ax.transAxes,color=colors[c],lw=2,clip_on=False)
 axes[r,0].text(-.11,.5,labels[r],transform=axes[r,0].transAxes,ha='right',va='center',fontsize=7.7,linespacing=1.3)
for ext in ['pdf','png','svg']:
 fig.savefig(root/f'figures/main/fig05_qualitative_results.{ext}',dpi=300)
plt.close(fig)
