"""Image-clustered contrasts and figures for the predeclared matched-mask experiment."""
import argparse,csv,html,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/'runs/matched_geometry_30k_v4'
COLORS={'large':'#BB6545','fragmented':'#267B9C'}
def read(p):return json.loads(Path(p).read_text())
def save(p,obj):Path(p).write_text(json.dumps(obj,indent=2,allow_nan=False)+'\n')
def csv_out(name,rows):
    keys=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w')as f:
        w=csv.DictWriter(f,keys);w.writeheader();w.writerows(rows)
def figure(fig,name):
    for ext in ['png','svg','pdf']:fig.savefig(OUT/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)
def interval(vals):
    vals=np.asarray(vals,dtype=float);rng=np.random.default_rng(202610221)
    boot=vals[rng.integers(len(vals),size=(10000,len(vals)))].mean(1)
    return {'mean':float(vals.mean()),'ci95':np.percentile(boot,[2.5,97.5]).tolist(),'images':len(vals)}
def masks_figure():
    directory=OUT/'mask_audit';audit=read(directory/'mask_audit.json')
    data=np.load(directory/'inputs_preview.npz');cov=np.load(directory/'coverage_preview.npz')
    fig,axs=plt.subplots(3,5,figsize=(12,7),layout='constrained')
    for i in range(3):
        rgb=data['dev'][i].transpose(1,2,0)/255
        a,b=data['dev_masks'][i,0]
        for j,(im,title)in enumerate([(rgb,'Truth'),(rgb*(1-a[:,:,None]),'Single large hole'),(rgb*(1-b[:,:,None]),'Four local holes'),(a.astype(int)-b.astype(int),'Pairwise mask difference')]):
            axs[i,j].imshow(im,cmap='RdBu_r' if j==3 else None,vmin=-1 if j==3 else None,vmax=1 if j==3 else None);axs[i,j].set_title(title if i==0 else '');axs[i,j].axis('off')
        if i<2:
            key=['large','fragmented'][i];im=axs[i,4].imshow(cov[key],vmin=0,vmax=1,cmap='viridis');axs[i,4].set_title(f'{key} dev coverage');axs[i,4].axis('off')
        else:
            axs[i,4].axis('off');d=audit['splits']['dev']
            axs[i,4].text(0,1,f"Dev: {d['retained_images']}/{d['images']} images\n4 pairs / image\n1638 missing pixels\nCenter16 pixels identical\nQuadrant areas identical\nCore delta: {d['mean_signed_core_difference']:+.1f} px\nHead delta: {d['mean_signed_head_difference']:+.1f} px\nCount calipers: <=32 px, <=3%",va='top',fontsize=9)
    fig.suptitle('Matched-mask audit | fixed first 3 retained dev images, pair 0 | annotations used only for mask selection')
    figure(fig,'mask_controls')
    before=read(OUT/'before_after_confound_audit.json');v=before['mean'];csv_out('confound_source_data.csv',before['rows'])
    fig,axs=plt.subplots(1,3,figsize=(11,3.3),layout='constrained')
    xx=np.arange(2)
    axs[0].bar(xx-.16,[v['old_center16_large']*100,audit['splits']['dev']['center16_coverage'][0]*100],width=.32,color=COLORS['large'],label='large')
    axs[0].bar(xx+.16,[v['old_center16_fragmented']*100,audit['splits']['dev']['center16_coverage'][1]*100],width=.32,color=COLORS['fragmented'],label='fragmented')
    axs[0].set_ylabel('Missing pixels in central16 (%)');axs[0].legend(fontsize=8)
    for ax,region in zip(axs[1:],['core','head']):
        vals=[v[f'{tag}_{region}_region_fraction_delta']*100 for tag in ['old','new']]
        ax.bar(xx,vals,color=['#A5ADB3','#397D9D'],width=.6);ax.set_ylabel(f'Mean {region} occlusion difference (pp)')
        for j,value in enumerate(vals):ax.text(j,value+.3,f'{value:.2f}',ha='center',fontsize=9)
    for ax in axs:ax.set_xticks(xx,['Old masks','Matched masks']);ax.set_title('Central position'if ax is axs[0]else 'Large minus fragmented');ax.set_ylim(bottom=0)
    fig.suptitle('Annotation-only confound audit | same 124 original dev images | residual count differences remain')
    figure(fig,'confound_reduction')

def contrasts(rows,meta,seed):
    lookup={(r['geometry'],r['nfe'],r['image_index'],r['mask_pair']):r for r in rows}
    images=sorted(set(r['image_index']for r in rows));result=[];conditions=[]
    for i in images:
        rec=meta[i]
        for j in range(4):
            c=np.array(rec['large_counts'][j])-np.array(rec['fragmented_counts'][j])
            d={'seed':seed,'image_index':i,'original_index':rec['original_index'],'id':rec['id'],'pair':j,'core_delta':int(c[0]),'head_delta':int(c[3]),'possible_delta':int(c[2])}
            for aa,bb in [(4,16),(1,16),(4,8)]:
                for g in COLORS:d[f'G_{g}_{aa}_{bb}']=lookup[g,aa,i,j]['lpips_truth']-lookup[g,bb,i,j]['lpips_truth']
                d[f'H_{aa}_{bb}']=d[f'G_large_{aa}_{bb}']-d[f'G_fragmented_{aa}_{bb}']
            for metric,sign in [('sample_psnr',-1),('energy_score',1),('ensemble_variance',1)]:
                for g in COLORS:d[f'gain_{metric}_{g}_4_16']=sign*(lookup[g,4,i,j][metric]-lookup[g,16,i,j][metric])
            conditions.append(d)
    for a,b in [(4,16),(1,16),(4,8)]:
        for threshold in [None,8,16]:
            rr=[r for r in conditions if threshold is None or max(abs(r['core_delta']),abs(r['head_delta']))<=threshold]
            ix=sorted(set(r['image_index']for r in rr))
            if not ix:continue
            for quantity in [f'G_large_{a}_{b}',f'G_fragmented_{a}_{b}',f'H_{a}_{b}']:
                vals=[np.mean([r[quantity]for r in rr if r['image_index']==i])for i in ix]
                result.append({'seed':seed,'contrast':f'{a}to{b}','quantity':quantity,'subset':'all' if threshold is None else f'core_head_le{threshold}px','pairs':len(rr),**interval(vals)})
    # Cluster bootstrap resamples original images and retains all four rows/image.
    x=np.array([[1.,r['core_delta'],r['head_delta']]for r in conditions]);y=np.array([r['H_4_16']for r in conditions])
    coef=np.linalg.lstsq(x,y,rcond=None)[0];rng=np.random.default_rng(202610221)
    boot=[]
    groups=np.array([[j for j,r in enumerate(conditions)if r['image_index']==i]for i in images])
    for _ in range(10000):
        ix=groups[rng.integers(len(images),size=len(images))].ravel();boot.append(np.linalg.lstsq(x[ix],y[ix],rcond=None)[0][0])
    adjusted={'seed':seed,'contrast':'4to16','quantity':'H_OLS_intercept_at_exact_amount_match','subset':'linear_adjustment','mean':float(coef[0]),'ci95':np.percentile(boot,[2.5,97.5]).tolist(),'images':len(images),'pairs':len(conditions),'rank':int(np.linalg.matrix_rank(x)),'slopes_core_head':coef[1:].tolist(),'assumption':'linear residual count effects; descriptive extrapolation to zero count difference'}
    result.append(adjusted)
    for key in conditions[0]:
        if key.startswith('gain_'):
            vals=[np.mean([r[key]for r in conditions if r['image_index']==i])for i in images]
            result.append({'seed':seed,'contrast':'4to16','quantity':key,'subset':'all','pairs':len(conditions),**interval(vals)})
    return result,conditions

def main(audit_only=False):
    OUT.mkdir(exist_ok=True,parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,'legend.frameon':False})
    masks_figure()
    if audit_only:return
    verification=read(OUT/'verification.json');assert verification['passed']
    meta=read(OUT/'mask_audit/pair_metadata.json')['dev'];task=[];curves=[];loss=[];history=[];allcontrasts=[];allconditions=[]
    for seed in [0,1]:
        sd=OUT/f'seed{seed}';assert read(sd/'summary.json')['status']=='complete'
        rows=read(sd/'fm_step30000_final_rows.json');contr,cond=contrasts(rows,meta,seed);allcontrasts.extend(contr);allconditions.extend(cond)
        task.extend(read(sd/'fm_step30000_final_summary.json'))
        history.extend(json.loads(s)for s in (sd/'train.jsonl').read_text().splitlines())
        for step in [10000,20000,30000]:
            loss.extend(dict(seed=seed,**r)for r in read(sd/f'fm_step{step}_loss_summary.json'))
            curves.extend(read(sd/f'fm_step{step}_curve_summary.json'))
    save(OUT/'contrasts.json',allcontrasts);csv_out('paired_condition_contrasts.csv',allconditions)
    csv_out('task_source_data.csv',task);csv_out('loss_source_data.csv',loss);csv_out('training_source_data.csv',history)
    csv_out('checkpoint_task_source_data.csv',curves)
    csv_out('contrast_source_data.csv',[{**r,'ci95_low':r['ci95'][0],'ci95_high':r['ci95'][1]}for r in allcontrasts])
    fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for seed in [0,1]:
        for col,(metric,label)in enumerate([('lpips_truth','LPIPS to observed truth (lower better)'),('energy_score','Pixel energy score (lower better)')]):
            ax=axs[seed,col]
            for g,color in COLORS.items():
                rr=sorted([r for r in task if r['seed']==seed and r['geometry']==g],key=lambda r:r['nfe'])
                ax.plot([r['nfe']for r in rr],[r[metric]for r in rr],'-o',color=color,ms=4,label=g)
            ax.set_xscale('log',base=2);ax.set_xticks([1,2,4,8,16,32],[1,2,4,8,16,32]);ax.set_xlabel('Euler sampling steps / NFE');ax.set_ylabel(label);ax.set_title(f'Seed {seed} | 30k updates');ax.legend(fontsize=8)
    fig.suptitle(f'Controlled local geometry | {len(meta)} dev images x 4 mask pairs x 4 shared noise samples')
    figure(fig,'nfe_curves')
    fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    convergence=[]
    for seed in [0,1]:
        for g,c in COLORS.items():
            def value(step,nfe):return next(r['lpips_truth']for r in curves if r['seed']==seed and r['step']==step and r['nfe']==nfe and r['geometry']==g)
            for nfe,style in [(4,'-'),(16,'--')]:axs[seed,0].plot([10,20,30],[value(s,nfe)for s in [10000,20000,30000]],style,color=c,marker='o',label=f'{g}, NFE{nfe}')
            gains=[value(s,4)-value(s,16)for s in [10000,20000,30000]]
            axs[seed,1].plot([10,20,30],gains,'-o',color=c,label=g)
            convergence.append({'seed':seed,'geometry':g,'pair0_gain_4to16_at10k20k30k':gains,'LPIPS4_improvement_20kto30k':value(20000,4)-value(30000,4),'LPIPS16_improvement_20kto30k':value(20000,16)-value(30000,16)})
        axs[seed,0].set_ylabel('LPIPS to observed truth (lower better)');axs[seed,1].set_ylabel('G = LPIPS4 - LPIPS16');axs[seed,1].axhline(0,lw=.8,color='#777777')
        for ax in axs[seed]:ax.set_xlabel('Optimizer updates (thousands)');ax.set_title(f'Seed {seed}');ax.legend(fontsize=7)
    fig.suptitle('Fixed task validation curves | same pair 0 and noise at every checkpoint | final main test uses 4 pairs')
    figure(fig,'task_validation_curves');save(OUT/'convergence.json',convergence)
    fig,axs=plt.subplots(1,3,figsize=(12,3.7),sharey=True,layout='constrained')
    for seed,marker in [(0,'o'),(1,'s')]:
        for j,(subset,label)in enumerate([('all','All matched pairs'),('core_head_le8px','Core/head differences <=8 px'),('core_head_le16px','Core/head differences <=16 px')]):
            r=next(r for r in allcontrasts if r['seed']==seed and r['quantity']=='H_4_16'and r['subset']==subset)
            axs[j].errorbar(seed,r['mean'],yerr=np.array([[r['mean']-r['ci95'][0]],[r['ci95'][1]-r['mean']]]),fmt=marker,capsize=4,color='#334B5D');axs[j].set_title(label+f"\nn={r['images']} images, {r['pairs']} pairs");axs[j].axhline(0,color='#777777',lw=.8);axs[j].set_xticks([0,1],['Seed 0','Seed 1']);axs[j].set_xlim(-.5,1.5)
            if j==0:axs[j].set_ylabel('H = extra-step gain, large minus fragmented')
    fig.suptitle('4 to 16 steps | positive H favors large holes | 95% paired image-bootstrap intervals')
    figure(fig,'interaction_sensitivity')
    fig,axs=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for seed in [0,1]:
        ax=axs[seed,0]
        for g,c in COLORS.items():
            for split,style in [('train_probe','--'),('dev','-')]:
                yy=[np.mean([r['loss']for r in loss if r['seed']==seed and r['step']==step and r['variant']=='ema'and r['geometry']==g and r['split']==split])for step in [10000,20000,30000]]
                ax.plot([10,20,30],yy,style,color=c,marker='o',label=g+'/'+split)
        ax.set_title(f'Seed {seed}: fixed-probe velocity MSE');ax.set_xlabel('Optimizer updates (thousands)');ax.set_ylabel('Mean over five equal time bins');ax.legend(fontsize=7)
        ax=axs[seed,1]
        for g,c in COLORS.items():
            for step,alpha in [(10000,.35),(20000,.65),(30000,1.)]:
                rr=sorted([r for r in loss if r['seed']==seed and r['step']==step and r['variant']=='ema'and r['geometry']==g and r['split']=='dev'],key=lambda r:r['t_low'])
                ax.plot([.1,.3,.5,.7,.9],[r['loss']for r in rr],'-o',color=c,alpha=alpha,label=f'{g} {step//1000}k')
        ax.set_title(f'Seed {seed}: dev time-bin losses');ax.set_xlabel('Interpolation time t');ax.set_ylabel('Masked velocity MSE');ax.legend(fontsize=6,ncol=2)
    figure(fig,'validation_losses')
    fig,axs=plt.subplots(2,7,figsize=(13,4.4),layout='constrained')
    data=np.load(OUT/'mask_audit/inputs_preview.npz');examples=np.load(OUT/'seed0/fm_step30000_final_examples.npz');rgb=data['dev'][0].transpose(1,2,0)/255
    for gi,g in enumerate(COLORS):
        axs[gi,0].imshow(rgb*(1-data['dev_masks'][0,0,gi,:,:,None]));axs[gi,0].set_title(g+' input',fontsize=8);axs[gi,0].axis('off')
        for j,nfe in enumerate([1,2,4,8,16,32],1):
            axs[gi,j].imshow((examples[f'{g}_{nfe}'][0,0].transpose(1,2,0)+1)/2);axs[gi,j].set_title(f'NFE {nfe}',fontsize=8);axs[gi,j].axis('off')
    fig.suptitle('Fixed example: seed 0, first retained dev image, mask pair 0, noise 0 (not selected by quality)');figure(fig,'fixed_example')
    criterion=[]
    for seed in [0,1]:
        def pick(q,subset='all'):return next(r for r in allcontrasts if r['seed']==seed and r['quantity']==q and r['subset']==subset)
        H=pick('H_4_16');G=pick('G_large_4_16');S=pick('H_4_16','core_head_le8px')
        criterion.append({'seed':seed,'H_positive':H['mean']>0,'large_actually_improves':G['mean']>0,'H_at_least_002':H['mean']>=.002,'large_gain_at_least_002':G['mean']>=.002,'tight_subset_direction_positive':S['mean']>0,'tight_subset_at_least30images':S['images']>=30})
    supported=all(all(v for k,v in r.items()if k!='seed')for r in criterion)
    tight_ci_positive=all(next(r for r in allcontrasts if r['seed']==seed and r['quantity']=='H_4_16'and r['subset']=='core_head_le8px')['ci95'][0]>0 for seed in [0,1])
    decision={'predeclared_point_estimate_rule_passed':supported,'strict_subset_positive_ci_both_seeds':tight_ci_positive,'per_seed':criterion,'training_updates_total':60000,'network_parameters':1747523,'interpretation':'Exploratory controlled local-mask result on common-support dev; two training seeds, no official test; boundary and semantic identities outside center16 still differ. Passing the predeclared directional/magnitude rule does not imply that tight-subset confidence intervals exclude zero.'}
    best=[{'seed':seed,'geometry':g,'best_scanned_nfe_by_lpips':min([r for r in task if r['seed']==seed and r['geometry']==g],key=lambda r:r['lpips_truth'])['nfe']}for seed in [0,1]for g in COLORS]
    decision['descriptive_best_nfe_scan']=best
    best_text='；'.join(f"seed{r['seed']} {r['geometry']}：{r['best_scanned_nfe_by_lpips']}步"for r in best)
    save(OUT/'decision.json',decision)
    table=[]
    for seed in [0,1]:
        row=[str(seed)]
        for q in ['G_large_4_16','G_fragmented_4_16','H_4_16']:
            r=next(r for r in allcontrasts if r['seed']==seed and r['quantity']==q and r['subset']=='all');row.append(f"{r['mean']:+.5f} [{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}]")
        table.append('<tr>'+''.join(f'<td>{v}</td>'for v in row)+'</tr>')
    secondary=[]
    for seed in [0,1]:
        for g in COLORS:
            cells=[str(seed),g]
            for metric in ['sample_psnr','energy_score']:
                r=next(r for r in allcontrasts if r['seed']==seed and r['quantity']==f'gain_{metric}_{g}_4_16')
                cells.append(f"{r['mean']:+.5f} [{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}]")
            secondary.append('<tr>'+''.join(f'<td>{v}</td>'for v in cells)+'</tr>')
    tight_text=[]
    for seed in [0,1]:
        r=next(r for r in allcontrasts if r['seed']==seed and r['quantity']=='H_4_16'and r['subset']=='core_head_le8px')
        tight_text.append(f"seed{seed}：{r['mean']:+.5f} [{r['ci95'][0]:+.5f}, {r['ci95'][1]:+.5f}]")
    audit=read(OUT/'mask_audit/mask_audit.json');dd=audit['splits']['dev']
    if supported:
        conclusion='两个种子通过预设方向和幅度判据：在这组匹配局部掩码上，大洞从4→16步有额外收益。'+('严格匹配子集的区间也均高于零。'if tight_ci_positive else '严格匹配子集的区间仍跨零，独立于残余遮挡差异的几何效应证据尚不充分。')
    elif all(r['H_positive'] and r['large_actually_improves'] for r in criterion):
        conclusion='两个种子的方向保留，但尚未满足预设实际意义或严格匹配稳健性判据；应收紧原来的几何主张。'
    elif all(r['H_positive']for r in criterion):
        conclusion='大洞与碎洞的相对收益差为正，但大洞本身并未在两个种子中都实际获益，原来的“更受益”表述未稳定复现。'
    else:
        conclusion='尚未在两个种子中稳定复现原来的几何现象；不能沿用旧实验结论。'
    body=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>GeoStep · 位置与前景匹配对照</title><style>body{{font:16px/1.7 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;max-width:1120px;margin:40px auto;padding:0 24px;color:#20313b}}img{{width:100%}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:12px;border-bottom:1px solid #ccd5da;text-align:left}}.card{{background:#edf3f6;padding:18px;border-radius:12px}}a{{color:#267b9c}}small{{color:#52646f}}</style>
<h1>GeoStep：排除位置与前景遮挡的主要混杂后，现象是否保留？</h1><p class="card">{conclusion}本页只报告已完成且通过核验的两个30k训练，不将旧掩码成绩混为新对照。</p>
<h2>受控问题与证据范围</h2><p>同一个条件流匹配CNN同时学习两种掩码。网络仍为1,747,523参数，seed0和seed1各从头训练30,000步，沿用原学习率、AdamW、EMA和批大小64。验证固定为{len(meta)}张原dev图、每图4对掩码、每对4个共用独立噪声。主比较固定为4→16个Euler步。</p>
<p>大洞为31×53近矩形，去掉5个角部像素；碎洞为同一区域附近的4个局部洞，1像素可见切缝。两类均缺失1638像素。逐对中心16×16缺失像素完全相同、四象限缺失像素数量完全相同、质心偏差小于1像素。利用独立trimap和头框标注筛选掩码，并非输入模型。保留的训练图中7张缺少头框标注，未施加头部约束；验证图的头框标注齐全。前景、可能前景及头框遮挡差均≤区域面积3%且≤32像素。训练保留2812/2924，验证保留124/128；结果限定于共同支持子集。</p>
<img src="confound_reduction.png"><p>同样124张原图，旧对照的大洞平均多遮244.35个前景像素和145.85个头框像素；新对照分别降到10.76和6.40像素。旧中心16覆盖99.45%与39.08%，新两类均为76.26%。这些标注审计说明主要混杂减弱；掩码长宽比和碎洞布局同时改变，因而不能单凭新旧模型效应之差确定旧现象的唯一成因。</p><img src="mask_controls.png"><p><small>残余差异：dev大洞平均多遮前景{dd['mean_signed_core_difference']:.2f}像素、头框{dd['mean_signed_head_difference']:.2f}像素。全图覆盖并非逐像素相同；切缝和外缘差异属于本次干预。标注3表示未分类，未当成确定前景。</small></p>
<h2>预先固定的主要检验</h2><p>G = LPIPS(NFE4) − LPIPS(NFE16)，正数表示增加步数确实改善。H = G大洞 − G碎洞。H为正才表示大洞的额外收益更高；还必须检查G大洞本身为正，避免把“坏得较少”叫作受益。</p><table><thead><tr><th>种子</th><th>G大洞 [95% CI]</th><th>G碎洞 [95% CI]</th><th>H [95% CI]</th></tr></thead><tbody>{''.join(table)}</tbody></table><p>区间来自10,000次配对原图bootstrap，每张图先平均4个掩码和4个噪声。两个种子分别报告；这不是训练种子总体的置信区间。预设实际意义门槛要求两个种子的大洞收益G及相对收益差H均≥0.002。</p><p>严格前景/头框差≤8像素子集的H [95% CI]：{'；'.join(tight_text)}。更严格的匹配改变了验证子群、也减少了原图数，效应缩小与区间变宽不能单独归因于混杂消除。</p>
<img src="nfe_curves.png"><p>描述性步数扫描的最佳LPIPS：{best_text}。相对的4→16步收益差不等于两类掩码应采用不同的最佳步数，本轮未建立几何自适应采样优于统一步数的证据。</p><table><thead><tr><th>种子</th><th>几何</th><th>PSNR16−PSNR4 [95% CI]</th><th>Energy4−Energy16 [95% CI]</th></tr></thead><tbody>{''.join(secondary)}</tbody></table><p>两列均为正数更好：PSNR为遮挡区单样本PSNR的平均，energy score以遮挡区RMS距离计算、使用4个独立样本。它们与LPIPS衡量不同性质，不能将任一指标的改善概括成所有意义上的质量提升。</p><img src="interaction_sensitivity.png"><p>前景/头框差≤8像素和≤16像素为事先固定的敏感性子集；按原图聚类重采样。线性调整结果另存于contrasts.json，仅作为残余计数效应呈线性的描述性诊断。</p>
<h2>训练与数值核验</h2><img src="task_validation_curves.png"><p>上图的训练曲线始终使用每张图的同一掩码pair0，避免更换评估条件造成假趋势；最终主比较扩展到事前固定的全部4对掩码。</p><img src="validation_losses.png"><img src="fixed_example.png"><p>检查两种子各30k真实AdamW更新、全部检查点有限性、网络形状和来源、共享掩码、独立初始化，以及选定loss的独立NumPy重算和已知像素保持。<a href="verification.json">核验记录</a> · <a href="decision.json">预设判据逐项结果</a> · <a href="contrasts.json">全部比较和敏感性结果</a> · <a href="paired_condition_contrasts.csv">逐条件源数据</a></p>
<h2>能支持到哪里</h2><p>本轮检验的是相近位置、近似等前景遮挡的一个局部大洞与四个局部小洞；它改变了旧实验的长宽比和16孔布局。即使现象保留，也不能直接声称对任意掩码成立。LPIPS衡量与单张真值的配对距离，未证明样本分布真实性或语义不确定性。验证集已用于探索，GeoStep本轮未访问正式测试图像。若主判据不成立，应收缩或放弃“大洞普遍更需要更多步”的叙述，而不追加预算追逐正结果。</p><p><a href="../../matched_mask_protocol.json">掩码与训练方案</a> · <a href="../../matched_analysis_plan.json">训练前分析计划</a> · <a href="mask_audit/mask_audit.json">匹配审计</a></p></html>'''
    (OUT/'RESULTS_zh.html').write_text(body)
    print(json.dumps({'decision':decision,'primary':[r for r in allcontrasts if r['contrast']=='4to16']},ensure_ascii=False))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--audit-only',action='store_true');main(p.parse_args().audit_only)
