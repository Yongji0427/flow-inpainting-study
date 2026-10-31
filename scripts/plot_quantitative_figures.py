"""Reproduce manuscript quantitative figures from archived CSVs; no inference.
Run: python scripts/plot_quantitative_figures.py
Dependencies: numpy, matplotlib. Outputs PDF, SVG, PNG and plotted-coordinate audit.
"""
from pathlib import Path
import csv, json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
plt.rcParams['font.family']='sans-serif'
plt.rcParams['font.sans-serif']=['Arial','DejaVu Sans','Liberation Sans']
plt.rcParams['svg.fonttype']='none'
ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/'results/development'
OUT=Path(os.environ.get('FIGURE_OUTPUT_ROOT',str(ROOT/'figures')))
FM='#EA4335'; MSE='#4285F4'; PERCEPTUAL='#0F9D58'
LIGHT={'fm':'#F28E85','regression':'#8AB4F8'}
COLORS={'fm':FM,'regression':MSE,'perceptual_regression':PERCEPTUAL}
NFE=[1,2,4,8,16,32]; GEOMETRIES=['large','fragmented']; SEEDS=[0,1]; STEPS=[10000,20000,30000]
AUDIT={}
def read_csv(name):
    rows=list(csv.DictReader((DATA/name).open()))
    for r in rows:
        for k,v in r.items():
            try:r[k]=float(v) if any(c in v.lower() for c in ['.','e']) else int(v)
            except ValueError:pass
    return rows

def save(fig,name,directory='main',tight=False):
    target=OUT/directory; target.mkdir(parents=True,exist_ok=True)
    AUDIT[directory+'/'+name]=[{'xscale':ax.get_xscale(),'yscale':ax.get_yscale(),
        'lines':[{'x':np.asarray(line.get_xdata(),dtype=float).tolist(),'y':np.asarray(line.get_ydata(),dtype=float).tolist()} for line in ax.lines]} for ax in fig.axes]
    for ext in ['pdf','svg','png']:
        fig.savefig(target/f'{name}.{ext}',dpi=300,**({'bbox_inches':'tight'} if tight else {}))
    plt.close(fig)

def main_style():
    plt.rcParams.update({'font.size':7.1,'axes.titlesize':8,'axes.labelsize':7.4,
        'xtick.labelsize':6.7,'ytick.labelsize':6.7,'legend.fontsize':6.9,
        'axes.spines.top':True,'axes.spines.right':True,'axes.linewidth':.65,
        'xtick.major.size':2.5,'ytick.major.size':2.5,'xtick.major.width':.65,
        'ytick.major.width':.65,'xtick.direction':'in','ytick.direction':'in',
        'axes.edgecolor':'#444444','text.color':'#262626','axes.labelcolor':'#262626',
        'xtick.color':'#444444','ytick.color':'#444444','pdf.fonttype':42,'legend.frameon':False,'savefig.facecolor':'white'})

def panel(ax,letter,title):
    ax.set_title(f"$\\bf{{{letter}}}$  {title}",loc='left',pad=5)
    ax.grid(axis='y',color='#E8E8E8',linewidth=.55);ax.set_axisbelow(True)

def nfe_figure():
    rows=read_csv("figures_nfe_source.csv")

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 3.35))
    fig.subplots_adjust(left=.09, right=.987, bottom=.13, top=.85,
                        wspace=.35, hspace=.88)
    for gi, g in enumerate(GEOMETRIES):
        for mi, (metric, ylabel) in enumerate([
                ("lpips", "LPIPS ↓"), ("psnr_db", "Masked PSNR (dB) ↑")]):
            ax = axes[mi, gi]
            vals = []
            for seed in SEEDS:
                selected = sorted([r for r in rows if r["model"] == "fm" and
                                   r["geometry"] == g and r["seed"] == seed],
                                  key=lambda r: r["nfe"])
                assert [r["nfe"] for r in selected] == NFE
                y = [r[metric] for r in selected]
                vals.append(y)
            ax.plot(NFE, np.mean(vals, axis=0), color=FM, marker="o", markersize=3,
                    linewidth=1.6, zorder=4)
            for model, style in [("regression", "--"), ("perceptual_regression", ":")]:
                vals = [r[metric] for r in rows if r["model"] == model and r["geometry"] == g]
                assert len(vals) == 2
                ax.axhline(np.mean(vals), color=COLORS[model], linestyle=style,
                           linewidth=1.35, zorder=3)
            panel(ax, "abcd"[mi * 2 + gi], "Large hole" if g == "large" else "Fragmented mask")
            ax.set_xscale("log", base=2)
            ax.set_xlim(.88, 36)
            ax.set_xticks(NFE, NFE)
            ax.set_xlabel("FM network evaluations (NFE)")
            ax.set_ylabel(ylabel)
            ax.set_yticks([.10,.11,.12,.13] if mi==0 else [15.5,16,16.5,17])
            ax.set_ylim([[np.float64(0.0910335201315672), np.float64(0.13412962788603294)], [np.float64(0.0910335201315672), np.float64(0.13412962788603294)], [np.float64(15.2968245999467), np.float64(17.37254585710264)], [np.float64(15.2968245999467), np.float64(17.37254585710264)]][mi * 2 + gi])
    handles = [Line2D([], [], color=FM, marker="o", markersize=3, linewidth=1.6, label="FM"),
               Line2D([], [], color=MSE, linestyle="--", linewidth=1.35, label="MSE Regression (1 call)"),
               Line2D([], [], color=PERCEPTUAL, linestyle=":", linewidth=1.35,
                      label="Perceptual Regression (1 call)")]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(.53, .995),
               ncol=3, handlelength=2.6, columnspacing=1.5)
    save(fig, "fig03_nfe_quality")
    return rows

def training_figure():
    objective_rows=read_csv("figures_training_objectives.csv")
    aggregate=read_csv("figures_task_learning.csv")

    fig, axes = plt.subplots(2, 2, figsize=(6.5, 3.2))
    fig.subplots_adjust(left=.09, right=.987, bottom=.14, top=.91,
                        wspace=.25, hspace=.63)
    for j, (kind, title, ylabel) in enumerate([
            ("fm", "FM velocity objective", "Velocity MSE"),
            ("regression", "Regression image objective", "Image MSE")]):
        ax = axes[0, j]
        for split, style, marker in [("train_probe", "-", "o"), ("dev", "-", "s")]:
            vals = []
            for seed in SEEDS:
                y = [next(r["loss"] for r in objective_rows if r["model"] == kind and
                          r["seed"] == seed and r["step"] == step and r["split"] == split)
                     for step in STEPS]
                vals.append(y)
            ax.plot([10, 20, 30], np.mean(vals, axis=0), style, color=LIGHT[kind] if split=="train_probe" else COLORS[kind],
                    marker=marker, markersize=2.8, linewidth=1.4,
                    label="Train probe" if split == "train_probe" else "Dev")
        panel(ax, "ab"[j], title)
        ax.set_ylabel(ylabel)
        ax.set_xticks([10, 20, 30])
        ax.set_xlim(8.8, 31.2)
        ax.legend(loc="center right", fontsize=6.3, handlelength=2,
                  borderaxespad=.1, labelspacing=.25)

    for j, (metric, title, ylabel) in enumerate([
            ("lpips", "Dev perceptual quality", "LPIPS ↓"),
            ("psnr_db", "Dev pixel fidelity", "Masked PSNR (dB) ↑")]):
        ax = axes[1, j]
        for kind in ["fm", "regression"]:
            vals = []
            for seed in SEEDS:
                y = [next(r[metric] for r in aggregate if r["model"] == kind and
                          r["seed"] == seed and r["step"] == step) for step in STEPS]
                vals.append(y)
            ax.plot([10, 20, 30], np.mean(vals, axis=0), color=COLORS[kind],
                    marker="o" if kind == "fm" else "s", markersize=2.8,
                    linewidth=1.4, label="FM8" if kind == "fm" else "MSE Regression")
        panel(ax, "cd"[j], title)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Training updates (thousands)")
        ax.set_xticks([10, 20, 30])
        ax.set_xlim(8.8, 31.2)
    axes[1, 0].legend(loc="upper right", fontsize=6.3, handlelength=2,
                      borderaxespad=.1, labelspacing=.25)
    axes[1, 1].legend(loc="upper right", fontsize=6.3, handlelength=2,
                      borderaxespad=.4, labelspacing=.25)
    for ax, bounds in zip(axes.flat, [[np.float64(0.07729506236740294), np.float64(0.21025788293368938)], [np.float64(0.0008234690854593395), np.float64(0.12368321374823711)], [np.float64(0.09689854688626984), np.float64(0.1406248377613543)], [np.float64(15.93073762328394), np.float64(16.52469319585831)]]):
        ax.set_ylim(bounds)
    save(fig, "fig04_training_dynamics")

def supplementary_figures():
    plt.rcParams.update({'font.size':9,'axes.titlesize':10,'axes.labelsize':9,'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'axes.linewidth':.65})
    colors={'fm':FM,'regression':MSE,'large':FM,'fragmented':'#F28E85'}
    capacity=read_csv('capacity_seed0.csv'); task=read_csv('task_learning_pair0.csv')
    loss=read_csv('training_objectives.csv'); timebins=read_csv('fm_time_bins_30k.csv'); nfe=read_csv('nfe_sweep_dev.csv')
    def save_appendix(fig,name):save(fig,name,'supplementary',tight=True)
    fig,axs=plt.subplots(1,2,figsize=(7.0,2.5));metrics=[('lpips','LPIPS (lower is better)'),('psnr_db','Masked PSNR (dB)')]
    for ax,(metric,label)in zip(axs,metrics):
     for g in ['large','fragmented']:
      rr=[r for r in capacity if r['geometry']==g];ax.plot([r['parameters']/1e6 for r in rr],[r[metric]for r in rr],linestyle='-' if g=='large' else '--',marker='o' if g=='large' else 's',color=colors[g],label=g.capitalize())
     ax.set_xlabel('Trainable CNN parameters (millions)');ax.set_ylabel(label);ax.set_xticks([1.747523,6.734723,14.986435],['1.75','6.73','14.99']);ax.grid(axis='y',color='#E8E8E8',lw=.55);ax.set_axisbelow(True)
    axs[0].set_title('a  Fixed 30k updates, seed0',loc='left');axs[1].set_title('b  Same masks and NFE8',loc='left');axs[0].legend();axs[1].legend();fig.tight_layout();save_appendix(fig,'figS04_model_capacity')
    fig,axs=plt.subplots(2,2,figsize=(7.0,4.4))
    for gi,g in enumerate(['large','fragmented']):
     for mi,(metric,label)in enumerate(metrics):
      ax=axs[gi,mi]
      for kind in ['fm','regression']:
       seedvalues=[]
       for seed in [0,1]:
        rr=sorted([r for r in task if r['model']==kind and r['seed']==seed and r['geometry']==g],key=lambda r:r['step']);y=[r[metric]for r in rr];seedvalues.append(y)
       ax.plot([10,20,30],np.mean(seedvalues,axis=0),linestyle='-',marker='o' if kind=='fm' else 's',color=colors[kind],lw=1.8,label='FM8'if kind=='fm'else'Regression1')
      ax.set_title(f'{chr(97+gi*2+mi)}  {g.capitalize()}',loc='left');ax.set_xlabel('Training updates (thousands)');ax.set_ylabel(label);ax.set_xticks([10,20,30]);ax.grid(axis='y',color='#E8E8E8',lw=.55);ax.set_axisbelow(True)

    for ax,bounds in zip(axs.flat,[[np.float64(0.10555391964622803), np.float64(0.1509388241931916)], [np.float64(15.052916056110012), np.float64(15.896433808726648)], [np.float64(0.08817101862101305), np.float64(0.13031428730595976)], [np.float64(16.394450070396548), np.float64(17.236418633691724)]]):
     ax.set_ylim(bounds);ax.legend(fontsize=7,loc='best')
    fig.tight_layout();save_appendix(fig,'figS01_learning_curves')
    fig,axs=plt.subplots(1,2,figsize=(7.0,2.8))
    for ax,kind,title in zip(axs,['fm','regression'],['a  FM velocity objective','b  Regression image objective']):
     for split,c,label in [('train_probe',LIGHT[kind],'Train probe (128 images)'),('dev',COLORS[kind],'Dev (124 images)')]:
      for variant,style in [('online','--'),('ema','-')]:
       vals=[]
       for step in [10000,20000,30000]:vals.append(np.mean([r['loss']for r in loss if r['model']==kind and r['split']==split and r['variant']==variant and r['step']==step]))
       ax.plot([10,20,30],vals,style,color=c,marker='o' if split=='train_probe' else 's',markersize=3,label=f'{label}, {variant.upper()}')
     ax.set_title(title,loc='left');ax.set_xlabel('Training updates (thousands)');ax.set_ylabel('Fixed-probe objective');ax.set_xticks([10,20,30]);ax.grid(axis='y',color='#E8E8E8',lw=.55);ax.set_axisbelow(True)
    fig.legend(handles=[Line2D([],[],color='#999999' if split=='Train probe' else '#444444',linestyle=style,marker='o' if split=='Train probe' else 's',markersize=3,label=f'{split}, {variant}') for split in ['Train probe','Dev'] for variant,style in [('ONLINE','--'),('EMA','-')]],loc='upper center',bbox_to_anchor=(.5,1.12),ncol=2,fontsize=7.6);fig.tight_layout();save_appendix(fig,'figS02_objective_diagnostics')
    fig,axs=plt.subplots(1,2,figsize=(7.0,2.6))
    for ax,g,title in zip(axs,['large','fragmented'],['a  Large','b  Fragmented']):
     for split,c,label in [('train_probe',LIGHT['fm'],'Train probe'),('dev',FM,'Dev')]:
      vals=[]
      for low in [0,.2,.4,.6,.8]:vals.append(np.mean([r['velocity_mse']for r in timebins if r['geometry']==g and r['split']==split and r['t_low']==low]))
      ax.plot([.1,.3,.5,.7,.9],vals,linestyle='-',marker='o' if split=='train_probe' else 's',color=c,label=label)
     ax.set_title(title,loc='left');ax.set_xlabel('FM time-bin midpoint');ax.set_ylabel('EMA velocity MSE');ax.grid(axis='y',color='#E8E8E8',lw=.55);ax.set_axisbelow(True)
    axs[0].legend();axs[1].legend();fig.tight_layout();save_appendix(fig,'figS03_time_bin_diagnostics')
    fig,axs=plt.subplots(1,2,figsize=(7.0,2.9))
    for ax,(metric,label)in zip(axs,metrics):
     for g in ['large','fragmented']:
      seedvalues=[]
      for seed in [0,1]:
       rr=sorted([r for r in nfe if r['model']=='fm'and r['seed']==seed and r['geometry']==g],key=lambda r:r['nfe']);y=[r[metric]for r in rr];seedvalues.append(y)
      ax.plot([1,2,4,8,16,32],np.mean(seedvalues,axis=0),linestyle='-' if g=='large' else '--',marker='o' if g=='large' else 's',color=FM,label=f'FM, {g}')
      reg=np.mean([r[metric]for r in nfe if r['model']=='regression'and r['geometry']==g]);ax.axhline(reg,color=MSE,ls='-' if g=='large' else '--',lw=1,label=f'Regression, {g}')
     ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16,32],[1,2,4,8,16,32]);ax.set_xlabel('FM function evaluations');ax.set_ylabel(label);ax.grid(axis='y',color='#E8E8E8',lw=.55);ax.set_axisbelow(True)

    for ax,bounds in zip(axs,[[np.float64(0.09106284056249404), np.float64(0.13351389883656894)], [np.float64(15.2968245999467), np.float64(17.37254585710264)]]):ax.set_ylim(bounds)
    axs[0].set_title('a  Perceptual quality',loc='left');axs[1].set_title('b  Pixel fidelity',loc='left');fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',bbox_to_anchor=(.5,1.10),ncol=2,fontsize=7.8);fig.tight_layout();save_appendix(fig,'figS05_nfe_quality')

if __name__=='__main__':
    main_style(); nfe_figure(); training_figure(); supplementary_figures()
    OUT.mkdir(parents=True,exist_ok=True)
    (OUT/'plotted_coordinates.json').write_text(json.dumps(AUDIT,indent=2)+'\n')
    print('Rendered seven quantitative figures from eight archived CSV files; no inference.')
