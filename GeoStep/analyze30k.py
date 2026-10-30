"""Plot fixed validation curves and assess whether a 50k extension has support."""
import argparse
import csv
import html
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


ROOT = Path(__file__).resolve().parent
RUN = ROOT/'runs/continuation_seed0_30k_v3'
COLORS = {'scattered':'#287C9E','large':'#BD6945'}
NFE_COLORS = {1:'#6C7886',4:'#377FA4',8:'#8772A5',16:'#BC6947'}
CHECKPOINTS = [10000,15000,20000,25000,30000]


def read(name):
    return json.loads((RUN/name).read_text())


def save_plot(fig,name):
    for ext in ['png','svg','pdf']:
        fig.savefig(RUN/f'{name}.{ext}',dpi=180,bbox_inches='tight')
    plt.close(fig)


def export_csv(name,rows):
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with (RUN/name).open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def paired_interval(values,denominator=1.):
    values=np.asarray(values)
    denominator=np.asarray(denominator)
    rng=np.random.default_rng(202610217)
    indices=rng.integers(len(values),size=(5000,len(values)))
    divisor=denominator[indices].mean(1) if denominator.ndim else denominator
    boot=values[indices].mean(1)/divisor
    return {'improvement':float(values.mean()/denominator.mean()),'paired_image_bootstrap_95ci':np.percentile(boot,[2.5,97.5]).tolist()}


def main(partial=False):
    cfg=read('protocol.json')
    task=[];loss=[];histories={};loss_rows={};task_rows={}
    for kind in ['regression','fm']:
        path=RUN/f'{kind}_train.jsonl'
        histories[kind]=[json.loads(s) for s in path.read_text().splitlines()] if path.exists() else []
        for step in CHECKPOINTS:
            stem=f'{kind}_step{step}'
            if (RUN/f'{stem}_dev_summary.json').exists():
                task.extend(read(f'{stem}_dev_summary.json'))
                task_rows[(kind,step)]=read(f'{stem}_dev_rows.json')
            if (RUN/f'{stem}_loss_summary.json').exists():
                loss.extend(read(f'{stem}_loss_summary.json'))
                loss_rows[(kind,step)]=read(f'{stem}_loss_rows.json')
    assert task and loss
    complete=all((k,30000) in task_rows for k in ['regression','fm'])
    if not partial:
        assert complete and read('verification.json')['passed']
    export_csv('task_source_data.csv',task)
    export_csv('loss_source_data.csv',loss)
    export_csv('training_source_data.csv',sum(histories.values(),[]))
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42,
                         'legend.frameon':False})
    def objective(kind,step,split,geometry,variant='ema'):
        rr=[r for r in loss if (r['model'],r['step'],r['split'],r['geometry'],r['variant'])==(kind,step,split,geometry,variant)]
        return np.mean([r['loss'] for r in rr]) if rr else np.nan
    def group(kind,step,geometry,nfe):
        return next((r for r in task if (r['model'],r['step'],r['geometry'],r['nfe'])==(kind,step,geometry,nfe)),None)
    # Keep losses in separate panels: their target definitions differ.
    fig,axes=plt.subplots(1,3,figsize=(13,3.8),layout='constrained')
    for ax,kind in zip(axes[:2],['regression','fm']):
        for geom,color in COLORS.items():
            for split,style in [('train_probe','--'),('dev','-')]:
                steps=[s for s in CHECKPOINTS if (kind,s) in loss_rows]
                ax.plot(np.array(steps)/1000,[objective(kind,s,split,geom) for s in steps],style,
                        color=color,marker='o',ms=3,label=f'{geom} / {split}')
        ax.set_title('Regression objective' if kind=='regression' else 'FM velocity objective, uniform-bin average')
        ax.set_xlabel('Cumulative optimizer updates (thousands)');ax.set_ylabel('Fixed-probe masked MSE, EMA')
    axes[0].legend(fontsize=7,loc='upper center',bbox_to_anchor=(.5,-.24),ncol=2)
    for kind,style in [('regression','-'),('fm','--')]:
        rr=histories[kind]
        axes[2].plot([cfg['initial_step']/1000]+[r['step']/1000 for r in rr],
                     [cfg['learning_rate_start']]+[r['learning_rate'] for r in rr],style,label=kind)
    axes[2].set_title('Learning-rate intervention');axes[2].set_xlabel('Cumulative updates (thousands)')
    axes[2].set_ylabel('AdamW learning rate');axes[2].ticklabel_format(axis='y',style='sci',scilimits=(0,0));axes[2].legend(fontsize=8)
    for ax,label in zip(axes,'abc'):ax.text(-.12,1.04,label,transform=ax.transAxes,fontweight='bold')
    fig.suptitle('GeoStep | original CNN and masks | fixed train128 / dev128 probes | one seed'+(' | partial' if not complete else ''))
    save_plot(fig,'learning_curves')
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    fm_steps=[s for s in CHECKPOINTS if ('fm',s) in loss_rows]
    shades=['#ADB5BD','#9AB9CC','#709CB9','#477D9F','#1B516F']
    centers=np.arange(.1,1,.2)
    for column,geom in enumerate(COLORS):
        for s,color in zip(CHECKPOINTS,shades):
            if s not in fm_steps:continue
            rr=[r for r in loss if r['model']=='fm' and r['step']==s and r['geometry']==geom and r['variant']=='ema']
            def bins(split):return [next(r['loss'] for r in rr if r['split']==split and abs(r['t_low']-j/5)<1e-8) for j in range(5)]
            dv=np.array(bins('dev'));tr=np.array(bins('train_probe'))
            axes[0,column].plot(centers,dv,'-o',ms=3,color=color,label=f'{s//1000}k')
            axes[1,column].plot(centers,dv-tr,'-o',ms=3,color=color,label=f'{s//1000}k')
        axes[0,column].set_title(geom.capitalize());axes[0,column].set_ylabel('Dev velocity MSE, EMA')
        axes[1,column].set_ylabel('Dev minus fixed-train-probe velocity MSE')
        axes[1,column].axhline(0,color='#999999',lw=.7)
        for ax in axes[:,column]:
            ax.set_xlabel('Time interval');ax.set_xticks(centers,['0–.2','.2–.4','.4–.6','.6–.8','.8–1'])
    if fm_steps:axes[0,0].legend(ncol=3,fontsize=8)
    fig.suptitle('Fixed independent noise and stratified times | 128 images per split and geometry')
    save_plot(fig,'time_bin_loss')
    fig,axes=plt.subplots(2,3,figsize=(13,7),layout='constrained')
    for row,geom in enumerate(COLORS):
        for column,(metric,label) in enumerate([('sample_psnr','Sample PSNR (dB, higher)'),('lpips_truth','Paired LPIPS (lower)'),('energy_score','Pixel energy score (lower)')]):
            ax=axes[row,column]
            for kind,nfe,color in [('regression',1,'#333333')]+[('fm',n,NFE_COLORS[n]) for n in [1,4,16]]:
                entries=[group(kind,s,geom,nfe) for s in CHECKPOINTS]
                entries=[r for r in entries if r is not None]
                ax.plot([r['step']/1000 for r in entries],[r[metric] for r in entries],'-s' if kind=='regression' else '-o',
                        color=color,ms=3,label='Regression' if kind=='regression' else f'FM {nfe} NFE')
            ax.set_title(geom.capitalize());ax.set_xlabel('Cumulative updates (thousands)');ax.set_ylabel(label)
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='outside lower center',ncol=4,fontsize=8)
    fig.suptitle('Dev128 task curves | EMA | four fixed iid samples per condition | one seed')
    save_plot(fig,'task_validation_curves')
    fig,axes=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for row,geom in enumerate(COLORS):
        for column,(metric,label) in enumerate([('lpips_truth','Paired LPIPS (lower)'),('sample_psnr','Sample PSNR (dB, higher)')]):
            ax=axes[row,column]
            for s,color in zip(CHECKPOINTS,shades):
                entries=[group('fm',s,geom,n) for n in cfg['nfe']]
                entries=[r for r in entries if r is not None]
                ax.plot([r['nfe'] for r in entries],[r[metric] for r in entries],'-o',ms=3,color=color,label=f'{s//1000}k FM')
            reg=group('regression',30000 if complete else 10000,geom,1)
            if reg:ax.axhline(reg[metric],color='#333333',ls='--',lw=1,label=f"{reg['step']//1000}k regression")
            ax.set_xscale('log',base=2);ax.set_xlim(.8,40);ax.set_xticks(cfg['nfe'],cfg['nfe']);ax.set_xlabel('Euler NFE')
            ax.set_ylabel(label);ax.set_title(geom.capitalize())
    axes[0,0].legend(fontsize=7,ncol=2)
    fig.suptitle('Sampling-step tradeoff across optimization checkpoints | dev128')
    save_plot(fig,'nfe_validation_curves')
    # Actual batch-loss histories are descriptive, not fixed-condition validation.
    fig,axes=plt.subplots(1,2,figsize=(10,3.4),layout='constrained')
    for ax,kind in zip(axes,['regression','fm']):
        records=[]
        for dirname in ['pilot_seed0_v1','followup_seed0_v2','continuation_seed0_30k_v3']:
            f=ROOT/'runs'/dirname/f'{kind}_train.jsonl'
            if f.exists():records.extend(json.loads(s) for s in f.read_text().splitlines())
        records=[r for r in records if r['step']>=10000]
        ax.plot([r['step']/1000 for r in records],[r.get('loss_mean_last200',r.get('loss_mean_last100')) for r in records],color='#447891',lw=1)
        ax.set_title(kind.capitalize());ax.set_xlabel('Cumulative updates (thousands)');ax.set_ylabel('Random-batch training loss')
        ax.set_ylim(bottom=0)
    fig.suptitle('10k continuation | different objectives: loss heights are not cross-comparable | 200-update means')
    save_plot(fig,'batch_training_curves')
    if not complete:
        print(json.dumps({'status':'partial','available_task_assessments':list(map(str,task_rows))}));return
    comparisons=[]
    def lossvec(kind,step,geom):
        rr=[r for r in loss_rows[(kind,step)] if r['variant']=='ema' and r['split']=='dev' and r['geometry']==geom]
        return np.array([np.mean([r['loss'] for r in rr if r['index']==i]) for i in range(128)])
    def taskvec(kind,step,geom,nfe,metric):
        rr=sorted((r for r in task_rows[(kind,step)] if r['geometry']==geom and r['nfe']==nfe),key=lambda r:r['index'])
        return np.array([r[metric] for r in rr])
    for before,after in [(10000,30000),(20000,25000),(25000,30000)]:
        for kind in ['regression','fm']:
            for geom in COLORS:
                old=lossvec(kind,before,geom);new=lossvec(kind,after,geom)
                comparisons.append({'model':kind,'geometry':geom,'before':before,'after':after,
                                    'metric':'relative_objective_reduction','nfe':None,'threshold':.01,
                                    **paired_interval(old-new,old)})
                for nfe in ([1] if kind=='regression' else [4,16]):
                    for metric,threshold,sign in [('lpips_truth',.002,1),('sample_psnr',.1,-1)]:
                        old=taskvec(kind,before,geom,nfe,metric);new=taskvec(kind,after,geom,nfe,metric)
                        comparisons.append({'model':kind,'geometry':geom,'before':before,'after':after,
                                            'metric':metric,'nfe':nfe,'threshold':threshold,
                                            **paired_interval(sign*(old-new))})
    decision={}
    for kind in ['regression','fm']:
        primary=[r for r in comparisons if r['model']==kind and r['before']>=20000 and (r['metric']!='sample_psnr' or kind=='regression')]
        last=[r for r in primary if r['before']==25000]
        learning=[r for r in last if r['improvement']>=r['threshold'] and r['paired_image_bootstrap_95ci'][0]>0]
        degradation=[r for r in last if r['improvement']<=-r['threshold'] and r['paired_image_bootstrap_95ci'][1]<0]
        plateau=all(r['paired_image_bootstrap_95ci'][1]<r['threshold'] for r in primary)
        status='continued_learning_signal' if learning else ('practical_plateau_at_this_precision' if plateau else 'inconclusive_precision_or_tradeoff')
        decision[kind]={'status':status,'last_interval_learning_endpoints':learning,
                        'last_interval_degradation_endpoints':degradation,
                        'recommend_50k_for_training_diagnostic':bool(learning),
                        'automatic_training_started':False}
    report={'status':'complete','decision':decision,'comparisons':comparisons,
            'limitations':cfg['limitations']+['dev128 reused for exploration; no final test',
              'unadjusted paired image bootstrap, one seed; not a global convergence proof',
              'LPIPS is paired perceptual distance; sample variation is not semantic correctness']}
    (RUN/'decision_50k.json').write_text(json.dumps(report,indent=2))
    bin_comparisons=[]
    for before,after in [(10000,30000),(25000,30000)]:
        for geom in COLORS:
            for j in range(5):
                def binvec(step):
                    rr=sorted((r for r in loss_rows[('fm',step)] if r['variant']=='ema' and r['split']=='dev' and r['geometry']==geom and abs(r['t_low']-j/5)<1e-8),key=lambda r:r['index'])
                    return np.array([r['loss'] for r in rr])
                old,new=binvec(before),binvec(after)
                bin_comparisons.append({'before':before,'after':after,'geometry':geom,'t_low':j/5,'t_high':(j+1)/5,
                                        'before_loss':float(old.mean()),'after_loss':float(new.mean()),
                                        **paired_interval(old-new,old)})
    (RUN/'time_bin_comparisons.json').write_text(json.dumps(bin_comparisons,indent=2))
    export_csv('time_bin_comparison_source_data.csv',bin_comparisons)
    step_interactions=[]
    for step in CHECKPOINTS:
        gains={g:taskvec('fm',step,g,4,'lpips_truth')-taskvec('fm',step,g,16,'lpips_truth') for g in COLORS}
        step_interactions.append({'step':step,'positive_means':'LPIPS reduction with 4 to 16 Euler steps',
                                  'scattered':paired_interval(gains['scattered']),
                                  'large':paired_interval(gains['large']),
                                  'large_minus_scattered':paired_interval(gains['large']-gains['scattered']),
                                  'causal_geometry_claim':False})
    (RUN/'geometry_step_interaction.json').write_text(json.dumps(step_interactions,indent=2))
    single_step_progress=[{'geometry':g,'before':25000,'after':30000,
                          'metric':'lpips_truth','nfe':1,'role':'secondary endpoint diagnostic, not primary stopping criterion',
                          **paired_interval(taskvec('fm',25000,g,1,'lpips_truth')-taskvec('fm',30000,g,1,'lpips_truth'))}
                         for g in COLORS]
    (RUN/'single_step_progress.json').write_text(json.dumps(single_step_progress,indent=2))
    # Endpoint gap: a diagnostic, not a ground-truth conditional-mean error.
    inputs=np.load(RUN/'dev_inputs_preview.npz')
    endpoint=read('endpoint_gap_summary.json')
    export_csv('endpoint_gap_source_data.csv',endpoint)
    # Fixed first three dev conditions, first pre-specified Gaussian draw.
    columns=[('truth','Original'),('masked','Observed'),('reg10','Reg 10k'),('reg30','Reg 30k'),
             ('fm4_10','FM4 10k'),('fm4_30','FM4 30k'),('fm16_10','FM16 10k'),('fm16_30','FM16 30k')]
    canvas=Image.new('RGB',(8*132+12,6*154+48),'white');draw=ImageDraw.Draw(canvas)
    for col,(_,label) in enumerate(columns):draw.text((12+132*col,12),label,fill='black')
    files={(k,s):np.load(RUN/f'{k}_step{s}_dev_examples.npz') for k in ['regression','fm'] for s in [10000,30000]}
    for gi,geom in enumerate(COLORS):
        for i in range(3):
            truth=inputs['truth'][i].astype(np.float32)/127.5-1;m=inputs[geom][i]
            arrays={'truth':truth,'masked':truth*(1-m),
                    'reg10':files[('regression',10000)][f'{geom}_1'][0,i],
                    'reg30':files[('regression',30000)][f'{geom}_1'][0,i],
                    'fm4_10':files[('fm',10000)][f'{geom}_4'][0,i],
                    'fm4_30':files[('fm',30000)][f'{geom}_4'][0,i],
                    'fm16_10':files[('fm',10000)][f'{geom}_16'][0,i],
                    'fm16_30':files[('fm',30000)][f'{geom}_16'][0,i]}
            y=40+(gi*3+i)*154
            draw.text((12,y),f'{geom} / dev index {i}',fill='black')
            for col,(key,_) in enumerate(columns):
                a=np.clip((arrays[key].transpose(1,2,0)+1)*127.5,0,255).astype('uint8')
                canvas.paste(Image.fromarray(a).resize((128,128),Image.Resampling.NEAREST),(12+132*col,y+16))
    canvas.save(RUN/'fixed_examples_10k_30k.png')
    summary=read('summary.json');verify=read('verification.json')
    table=[]
    for geom in COLORS:
        for kind,nfe in [('regression',1),('fm',1),('fm',4),('fm',8),('fm',16),('fm',32)]:
            a,b=group(kind,10000,geom,nfe),group(kind,30000,geom,nfe)
            table.append(f'<tr><td>{geom}</td><td>{kind} {nfe} NFE</td><td>{a["sample_psnr"]:.3f} → {b["sample_psnr"]:.3f}</td><td>{a["lpips_truth"]:.5f} → {b["lpips_truth"]:.5f}</td></tr>')
    status_labels={'continued_learning_signal':'末段仍有达到预设幅度的学习信号',
                   'practical_plateau_at_this_precision':'最近两个区间可辨识的收益低于预设幅度',
                   'inconclusive_precision_or_tradeoff':'末段未见达阈值的收益，但尚不能确立平台期，或存在指标取舍'}
    decisions=''.join(f'<p><b>{k}</b>：{status_labels[v["status"]]}；延长至50k用于训练诊断的建议：{"是" if v["recommend_50k_for_training_diagnostic"] else "否"}。</p>' for k,v in decision.items())
    interaction_rows=''.join(f'<tr><td>{r["step"]}</td><td>{r["scattered"]["improvement"]:+.5f}</td><td>{r["large"]["improvement"]:+.5f}</td><td>{r["large_minus_scattered"]["improvement"]:+.5f}</td></tr>' for r in step_interactions)
    document=f'''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>GeoStep 30k 续训诊断</title>
<style>body{{max-width:1150px;margin:40px auto;font:16px/1.7 system-ui;color:#23313d;padding:0 24px}}img{{width:100%;height:auto}}table{{border-collapse:collapse;width:100%}}td,th{{padding:8px;border-bottom:1px solid #ddd}}pre{{white-space:pre-wrap;background:#f4f6f8;padding:15px}}h2{{margin-top:40px}}</style>
<h1>GeoStep：10k → 30k 训练充分性诊断</h1><p>两模型各追加20,000次更新，总新增40,000次。网络1,747,523参数、64×64分辨率、原训练掩码及其随机流均保持。仅在dev128评价，没有使用eval256或最终测试集。</p>
<p>本轮是学习率恢复与延长训练的联合优化干预，不能单独归因为训练时长。训练耗时：{summary['training_seconds']}；总墙钟{summary['wall_seconds']:.1f}秒。身份与数值核验通过：{verify['passed']}。</p>
<h2>是否延长至50k</h2>{decisions}<p>阈值预先固定：目标loss相对1%、LPIPS绝对0.002、回归PSNR 0.1dB。区间为图像级配对bootstrap，未校正多重比较，且不覆盖训练种子变化。非显著不代表等效。建议是探索性诊断，不是全局收敛证明；尚未自动启动50k。</p>
<p>上述FM停止判断针对预先固定的4/16步主指标。补充诊断显示，单步FM的LPIPS在25k→30k仍下降：分散洞0.00205、大洞0.00331。因此不能宣称所有采样设置均已收敛；如果下一轮专门研究单步端点学习，应独立考虑延长FM。当前优先级建议是修正掩码位置/前景混杂，而非同时延长两模型。</p>
<h2>固定验证与训练子集的目标曲线</h2><img src="learning_curves.png"><p>图中展示EMA；CSV/JSON另保留online模型。训练子集128张，每个几何、每个时间区间使用固定独立噪声与分层采样时间。两种模型的loss定义不同，不直接比较其数值。</p>
<h2>FM分时间段loss</h2><img src="time_bin_loss.png"><p>上排：dev速度MSE。下排：dev与固定训练子集的差值。每个区间128张图、一个固定噪声/时间抽样；仅为学习诊断。</p>
<h2>固定NFE的验证质量曲线</h2><img src="task_validation_curves.png"><p>四个预先固定的iid生成噪声，所有检查点共享。LPIPS为真值配对感知距离，energy score为像素空间分布评分；两者都不证明语义正确或真实后验。</p>
<h2>NFE曲线是否随训练改变</h2><img src="nfe_validation_curves.png"><table><tr><th>掩码</th><th>方法</th><th>PSNR 10k→30k</th><th>LPIPS 10k→30k</th></tr>{''.join(table)}</table>
<h2>4→16步的探索性几何交互</h2><table><tr><th>训练更新</th><th>分散洞LPIPS降低量</th><th>大洞LPIPS降低量</th><th>大洞−分散洞</th></tr>{interaction_rows}</table><p>正值表示增加步数改善了配对LPIPS，负值表示恶化。图像配对区间见geometry_step_interaction.json；原掩码的位置和前景混杂尚未排除，不能解释为纯几何因果效应。</p>
<h2>实际训练批次loss</h2><img src="batch_training_curves.png"><p>展示从10k开始的续训，200次更新窗口均值；该曲线的随机批次不断变化。</p>
<h2>预先固定的前三张dev样例</h2><img src="fixed_examples_10k_30k.png"><p>未挑选样例；使用第一个固定高斯噪声，像素统一映射至0–255并最近邻放大，无局部图像修饰。</p>
<h2>限制</h2><ul>{''.join('<li>'+html.escape(x)+'</li>' for x in report['limitations'])}</ul>
<p>原大洞与分散洞的位置/前景遮挡混杂仍存在，因此本轮不能确立纯几何因果结论。</p>
<p>源数据：<a href="loss_source_data.csv">分段目标loss</a> · <a href="task_source_data.csv">验证任务指标</a> · <a href="training_source_data.csv">实际训练记录</a> · <a href="decision_50k.json">配对比较和50k判断</a> · <a href="verification.json">独立核验</a> · <a href="checkpoint_manifest.json">服务器完整检查点清单</a>。</p></html>'''
    (RUN/'RESULTS_zh.html').write_text(document)
    (RUN/'figure_qa.txt').write_text('Python/matplotlib only; editable SVG/PDF text; source data in task/loss/training CSV.\nImages: fixed first three dev indices, first Gaussian draw, uniform display conversion, no selected crops.\nAll curves: one training seed; dev128 and fixed train128; no final test. PNG exports require visual inspection.\n')
    print(json.dumps({'status':'complete','decision':decision},indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--partial',action='store_true');a=p.parse_args();main(a.partial)
