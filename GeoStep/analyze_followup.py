"""Analyze fixed exploratory follow-up; preserve all unfavorable metrics."""
import csv
import json
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/followup_seed0_v2'


def read(name):return json.loads((RUN/name).read_text())


def ci(values):
    a=np.asarray(values);rng=np.random.default_rng(723)
    b=a[rng.integers(len(a),size=(5000,len(a)))].mean(1)
    return {'mean':float(a.mean()),'paired_image_bootstrap_95ci':np.percentile(b,[2.5,97.5]).tolist()}


def main():
    summary=read('summary.json');assert read('verification.json')['passed']
    groups=[];rows=[]
    for step in [2000,10000]:
        for kind in ['regression','fm','iid_noise']:
            groups+=read(f'{kind}_step{step}_eval_summary.json')
            rows+=read(f'{kind}_step{step}_eval_rows.json')
    def group(model,geom,nfe,step=10000):
        return next(r for r in groups if (r['model'],r['geometry'],r['nfe'],r['step'])==(model,geom,nfe,step))
    def vector(model,geom,nfe,metric,step=10000):
        r=sorted((r for r in rows if (r['model'],r['geometry'],r['nfe'],r['step'])==(model,geom,nfe,step)),key=lambda r:r['index'])
        assert len(r)==256
        return np.array([r[metric] for r in r])
    comparisons=[]
    for geom in ['scattered','large']:
        for nfe in [4,16]:
            for baseline in ['regression','regression_iid_noise']:
                for metric in ['lpips_truth','energy_score','sample_mse']:
                    values=vector(baseline,geom,1,metric)-vector('fm',geom,nfe,metric)
                    comparisons.append({'geometry':geom,'fm_nfe':nfe,'baseline':baseline,'metric':metric,
                                        'sign':'positive favors FM (lower metric)',**ci(values)})
    interactions={}
    for metric in ['sample_mse','lpips_truth','energy_score','mean_prediction_mse','ensemble_variance']:
        gain={g:vector('fm',g,1,metric)-vector('fm',g,16,metric) for g in ['scattered','large']}
        interactions[metric]={'scattered_1_to_16_reduction':float(gain['scattered'].mean()),
             'large_1_to_16_reduction':float(gain['large'].mean()),'large_minus_scattered':ci(gain['large']-gain['scattered'])}
    (RUN/'paired_comparisons.json').write_text(json.dumps({'comparisons':comparisons,'interactions':interactions},indent=2))
    budget_4_to_8={g:ci(vector('fm',g,4,'lpips_truth')-vector('fm',g,8,'lpips_truth'))
                   for g in ['scattered','large']}
    (RUN/'exploratory_4_to_8_lpips.json').write_text(json.dumps(budget_4_to_8,indent=2))
    fields=sorted(set().union(*(r.keys() for r in groups)))
    with (RUN/'metric_source_data.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(groups)
    colors={'scattered':'#247B9E','large':'#B75A42'}
    labels={'scattered':'Scattered','large':'Large'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    fig,axes=plt.subplots(2,3,figsize=(14,8),layout='constrained')
    plot_specs=[('lpips_truth','LPIPS to ground truth (lower)'),('energy_score','Pixel energy score (lower)'),
                ('sample_psnr','Sample PSNR (dB, higher)'),('ensemble_variance','Within-ensemble pixel variance'),
                ('lpips_pairwise','Between-sample LPIPS (difference only)')]
    for axis,(metric,ylabel) in zip(axes.flat,plot_specs):
        for geom,color in colors.items():
            for step,style in [(2000,':'),(10000,'-')]:
                d=[group('fm',geom,k,step) for k in [1,2,4,8,16,32]]
                axis.plot([r['nfe'] for r in d],[r[metric] for r in d],style,color=color,
                          marker='o' if step==10000 else None,ms=3,label=f'{labels[geom]} / {step:,} updates')
            if metric in ['lpips_truth','energy_score','sample_psnr']:
                axis.axhline(group('regression',geom,1)[metric],color=color,ls='--',lw=1,alpha=.65)
            if metric=='energy_score':
                axis.axhline(group('regression_iid_noise',geom,1)[metric],color=color,ls='-.',lw=1,alpha=.7)
        axis.set_xscale('log',base=2);axis.set_xticks([1,2,4,8,16,32],[1,2,4,8,16,32]);axis.set_xlabel('Euler NFE');axis.set_ylabel(ylabel)
    axes[0,0].legend(fontsize=7)
    refs=read('reference_step10000.json')
    normalized_reference=[]
    for geom in ['scattered','large']:
        base={r['index']:r['rmse_to_128'] for r in refs if r['geometry']==geom and r['nfe']==1}
        for r in refs:
            if r['geometry']==geom:
                normalized_reference.append({**r,'fraction_of_1nfe_error':r['rmse_to_128']/max(base[r['index']],1e-8)})
    (RUN/'normalized_reference.json').write_text(json.dumps(normalized_reference,indent=2))
    geometry_ref={}
    for nfe in [4,16,32]:
        arrays={g:np.array([r['fraction_of_1nfe_error'] for r in sorted(normalized_reference,key=lambda r:r['index'])
                          if r['geometry']==g and r['nfe']==nfe]) for g in ['scattered','large']}
        geometry_ref[str(nfe)]={'scattered_mean_fraction':float(arrays['scattered'].mean()),
                               'large_mean_fraction':float(arrays['large'].mean()),
                               'large_minus_scattered':ci(arrays['large']-arrays['scattered'])}
    (RUN/'normalized_reference_comparison.json').write_text(json.dumps(geometry_ref,indent=2))
    axis=axes[1,2]
    for geom,color in colors.items():
        d=[(k,np.mean([r['rmse_to_128'] for r in refs if r['geometry']==geom and r['nfe']==k])) for k in [1,2,4,8,16,32,64]]
        axis.plot([r[0] for r in d],[r[1] for r in d],'-o',ms=3,color=color,label=labels[geom])
    axis.axhline(.01,color='#777777',ls='--',lw=.8);axis.set_xscale('log',base=2);axis.set_yscale('log')
    axis.set_xticks([1,4,16,64],[1,4,16,64]);axis.set_xlabel('Euler NFE');axis.set_ylabel('RMSE to 128-NFE reference');axis.legend(fontsize=8)
    for axis,label in zip(axes.flat,'abcdef'):axis.text(-.14,1.04,label,transform=axis.transAxes,fontweight='bold',fontsize=12)
    fig.suptitle('GeoStep follow-up | one seed, 256 exploratory validation images, four fixed noise samples',fontsize=12)
    fig.savefig(RUN/'followup_results.png',dpi=180);fig.savefig(RUN/'followup_results.svg');fig.savefig(RUN/'followup_results.pdf');plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.4),layout='constrained')
    for axis,geom in zip(axes,['scattered','large']):
        for model,marker,color,name in [('fm','o',colors[geom],'FM'),('regression','s','#333333','Regression'),
                                       ('regression_iid_noise','X','#8B7395','Regression + iid noise')]:
            d=[group(model,geom,k) for k in ([1,2,4,8,16,32] if model=='fm' else [1])]
            axis.plot([r['sample_mse'] for r in d],[r['lpips_truth'] for r in d],marker=marker,
                      ls='-' if model=='fm' else 'none',color=color,label=name)
            if model=='fm':
                for r in d:
                    if r['nfe'] in [1,4,16,32]:axis.annotate(str(r['nfe']),(r['sample_mse'],r['lpips_truth']),xytext=(3,3),textcoords='offset points',fontsize=7)
        axis.set_title(labels[geom]);axis.set_xlabel('Missing-region sample MSE (lower)');axis.set_ylabel('LPIPS to ground truth (lower)');axis.legend(fontsize=7)
    fig.suptitle('Paired distortion / perceptual distance; neither axis establishes distribution realism',fontsize=10)
    fig.savefig(RUN/'quality_tradeoff.png',dpi=180);fig.savefig(RUN/'quality_tradeoff.svg');plt.close(fig)
    # Fixed first three images; originals and masks are taken from the preserved v1 plate.
    plate=Image.open(ROOT/'runs/pilot_seed0_v1/examples.png').convert('RGB')
    def example_archive(kind):
        full=RUN/f'{kind}_step10000_eval_examples.npz'
        return np.load(full if full.exists() else RUN/f'{kind}_step10000_eval_preview.npz')
    reg=example_archive('regression');fm=example_archive('fm')
    canvas=Image.new('RGB',(864,34+6*112+25),'white');draw=ImageDraw.Draw(canvas)
    headers=['Original','Masked','Regression','FM 1 / s0','FM 4 / s0','FM 16 / s0','FM 16 / s1']
    for j,h in enumerate(headers):draw.text((84+j*112,8),h,fill='black')
    for gi,geom in enumerate(['scattered','large']):
        draw.text((3,40+gi*3*112),labels[geom],fill='black')
        for i in range(3):
            row=gi*3+i;source_row=gi*8+i
            for j in range(2):
                tile=plate.crop((j*112,34+source_row*112,(j+1)*112,34+(source_row+1)*112))
                canvas.paste(tile,(80+j*112,34+row*112))
            arrays=[reg[f'{geom}_1'][0,i],fm[f'{geom}_1'][0,i],fm[f'{geom}_4'][0,i],
                    fm[f'{geom}_16'][0,i],fm[f'{geom}_16'][1,i]]
            for j,a in enumerate(arrays,2):
                rgb=((np.clip(a,-1,1).transpose(1,2,0)+1)*127.5).astype(np.uint8)
                tile=Image.fromarray(rgb).resize((104,104),Image.Resampling.NEAREST)
                canvas.paste(tile,(84+j*112,34+row*112))
    draw.text((4,34+6*112+5),'Fixed first 3 images; no outcome-based example selection. s0/s1 are fixed noise draws.',fill='black')
    canvas.save(RUN/'examples_followup.png')
    table=['| 遮挡 | 模型 | PSNR ↑ | LPIPS ↓ | Energy score ↓ | 样本均值 MSE ↓ | 样本方差 |',
           '|---|---|---:|---:|---:|---:|---:|']
    for geom in ['scattered','large']:
        for model,nfe,name in [('regression',1,'回归'),('fm',1,'FM 1 NFE'),('fm',4,'FM 4 NFE'),
                               ('fm',16,'FM 16 NFE'),('regression_iid_noise',1,'回归＋独立噪声')]:
            r=group(model,geom,nfe)
            table.append(f'| {"分散" if geom=="scattered" else "大块"} | {name} | {r["sample_psnr"]:.3f} | {r["lpips_truth"]:.4f} | {r["energy_score"]:.4f} | {r["mean_prediction_mse"]:.5f} | {r["ensemble_variance"]:.5f} |')
    decomp=[]
    for geom in ['scattered','large']:
        a,b=group('fm',geom,1),group('fm',geom,16)
        dm=b['sample_mse']-a['sample_mse'];dv=b['ensemble_variance']-a['ensemble_variance']
        decomp.append({'geometry':geom,'sample_mse_change_1_to_16':dm,
                       'mean_prediction_mse_change':b['mean_prediction_mse']-a['mean_prediction_mse'],
                       'variance_change':dv,'variance_fraction_of_mse_change':dv/dm if abs(dm)>1e-10 else None})
    (RUN/'error_decomposition.json').write_text(json.dumps(decomp,indent=2))
    lines=['# GeoStep 第二轮：评价修正与追加训练结果','',
       '日期：2026-10-21。本轮是看过第一轮结果后的探索性追加，不是确认性测试。','',
       f'已完成回归与FM各从2,000至10,000步的追加优化，新增16,000次更新，累计主训练20,000次。主过程用时{summary["wall_seconds"]:.1f}秒。两个模型的完整图片/掩码/翻转随机流哈希一致；独立数值复核通过。','',
       '## 是否值得继续','',
       '**建议继续一轮位置/内容匹配的关键对照；当前不宜直接扩成原创方法。** 同样缺失面积下，两类遮挡里的模型相对表现不同；掩码位置和遮住的内容尚未充分匹配，所以不能将其因果归于纯几何。',
       '',
       f'最终大块遮挡中，FM4的LPIPS为{group("fm","large",4)["lpips_truth"]:.4f}，回归为{group("regression","large",1)["lpips_truth"]:.4f}；分散遮挡中则为{group("fm","scattered",4)["lpips_truth"]:.4f}对{group("regression","scattered",1)["lpips_truth"]:.4f}。PSNR仍是回归更高。因此应研究几何如何改变重建—生成权衡，而不是把FM说成全面更优。',
       '',
       f'大块FM4与FM16的LPIPS分别为{group("fm","large",4)["lpips_truth"]:.4f}和{group("fm","large",16)["lpips_truth"]:.4f}，不能仅因为后者步数更多就推定更值得计算。FM16的energy score与独立噪声对照也没有呈现明确全面优势；相关配对区间在下文完整报告。',
       '',
       '下一轮若继续，优先验证“在缺失面积、位置分布和前景遮挡程度尽量匹配后，遮挡空间组织是否仍会改变确定性重建与少步条件生成的失真—感知—多样性权衡”。先排除位置/内容混杂并复现交互，再考虑根据掩码在回归和少步FM之间选择；本轮没有训练或验证该选择策略，也没有证明论文级新颖性。',
       '',
       f'当前验证网格中，分散遮挡LPIPS最低在4 NFE（{group("fm","scattered",4)["lpips_truth"]:.4f}），大块最低在8 NFE（{group("fm","large",8)["lpips_truth"]:.4f}）。4→8 NFE的LPIPS下降分别为{budget_4_to_8["scattered"]["mean"]:.4f}与{budget_4_to_8["large"]["mean"]:.4f}。这是查看网格后的探索性观察，不用于冻结策略或证明最优预算；其原因仍受下述位置/内容混杂影响。',
       '',
       '## 重要混杂：掩码中心覆盖率不匹配','',
       f'掩码位置审计显示：大块掩码的中心8x8平均覆盖率为{read("mask_position_audit.json")["large"]["center_8x8_coverage"]:.1%}，分散掩码为{read("mask_position_audit.json")["scattered"]["center_8x8_coverage"]:.1%}；中心16x16则分别为{read("mask_position_audit.json")["large"]["center_16x16_coverage"]:.1%}和{read("mask_position_audit.json")["scattered"]["center_16x16_coverage"]:.1%}。缺失总面积相同并不消除这个偏差。',
       '',
       '原因包括大矩形在不越界条件下几乎必然覆盖图中心，以及分散掩码的固定网格布局。若主体集中在中心，两种条件还可能移除不同程度的语义内容。当前FM与回归在各自固定条件下的比较仍有效，但两个条件之间的差异尚不能用于识别纯几何效应。下一轮应利用前景信息分层匹配掩码、平衡质心/边界位置，并先验证像素覆盖分布；本轮没有执行这些新对照。',
       '',
       '## 核心问题的修正','',
       '由“大洞是否需要更多步才能更接近原图”，修正为“采样计算如何改变条件输出的均值、方差、感知距离与数值稳定性；几何如何影响这些关系，以及追加优化后是否稳定”。保留PSNR，避免只改用对方法有利的指标。','',
       '- PSNR/MSE：逐样本原图恢复能力；四个固定噪声均评价，没有best-of-4选优。',
       '- 样本均值误差与方差：精确满足平均样本MSE=样本均值MSE+样本方差。四样本均值不是真实条件均值，这不是无偏后验偏差/方差分解。',
       '- LPIPS：SqueezeNet、版本0.1、包0.1.4、原生64x64完整合成图。它是与一张真值的感知距离，不是分布真实性。',
       '- 样本间LPIPS：输出差异，不直接代表合理语义多样性。',
       '- Energy score：缺失区域的归一化欧氏距离评分，以四样本U-statistic估计；越低越好。其高维有限样本结果不能单独证明纹理/语义质量。',
       '- 独立噪声对照：尺度由FM 16-NFE四样本方差确定，不看真值。只在期望上匹配截断前有限样本方差，实际实现和截断后方差均另报。原中心化依赖噪声对照排除于本报告。','',
       '## 最终10,000步结果','', '\n'.join(table),'',
       '所有数字来自同一256张先导验证图片、四个固定噪声。与首轮原表的单噪声结果不能直接当成同一估计；本轮已重评2,000步旧检查点以形成匹配比较。','',
       '![主要结果](followup_results.png)','',
       '图a—e：点线为2,000步、实线为10,000步；a—c中的水平虚线为最终回归基线，b中水平点划线为独立噪声对照。图e只表示样本差异。图f使用64张图的第一个固定噪声比较128-NFE数值参考，参考输出不是真值。','',
       '![质量关系](quality_tradeoff.png)','',
       '上图中两个坐标都越低越好。噪声对照只用于机制排除，不能作为有语义的生成器。','',
       '## 误差分解','',
       '| 几何 | 1→16样本MSE变化 | 样本均值MSE变化 | 样本方差变化 | 方差变化占总变化 |',
       '|---|---:|---:|---:|---:|']
    for r in decomp:
        frac=r['variance_fraction_of_mse_change']
        lines.append(f'| {r["geometry"]} | {r["sample_mse_change_1_to_16"]:.6f} | {r["mean_prediction_mse_change"]:.6f} | {r["variance_change"]:.6f} | {frac:.1%} |')
    lines += ['',
       '这个比例只描述四个固定输出的代数变化；较大的方差项不能自动被解释为合理不确定性，可能也含错误细节。','',
       '## 配对比较与几何交互','',
       '完整值见 paired_comparisons.json。下面显示最终FM4-NFE相对直接回归的LPIPS下降（正值有利于FM），以及FM16-NFE相对独立噪声对照的energy score下降。区间为未校正的探索性配对图像bootstrap 95%区间，不覆盖训练种子或重新抽噪声的不确定性。','']
    for g in ['scattered','large']:
        for n,b,m in [(4,'regression','lpips_truth'),(16,'regression_iid_noise','energy_score')]:
            c=next(r for r in comparisons if (r['geometry'],r['fm_nfe'],r['baseline'],r['metric'])==(g,n,b,m))
            low,high=c['paired_image_bootstrap_95ci']
            lines.append(f'- {g}, FM{n} vs {b}, {m}：{c["mean"]:.6f}，区间 [{low:.6f}, {high:.6f}]。')
    for m in ['lpips_truth','energy_score']:
        r=interactions[m];i=r['large_minus_scattered'];lo,hi=i['paired_image_bootstrap_95ci']
        lines.append(f'- {m} 的1→16收益几何交互（大块减分散）：{i["mean"]:.6f}，区间 [{lo:.6f}, {hi:.6f}]。')
    lines+=['','## 数值误差的尺度控制','',
       '以下为探索性附加分析：把每张图对128-NFE参考的RMSE除以该图1-NFE的RMSE，避免仅因大洞输出变化幅度更大而宣称其收敛更慢。这仍是固定模型的数值行为，不是真实条件分布或语义质量的证据。','']
    for nfe,r in geometry_ref.items():
        cc=r['large_minus_scattered'];low,high=cc['paired_image_bootstrap_95ci']
        lines.append(f'- {nfe} NFE：分散残差比例 {r["scattered_mean_fraction"]:.4f}，大块 {r["large_mean_fraction"]:.4f}；配对差 {cc["mean"]:.4f}，区间 [{low:.4f}, {high:.4f}]。')
    lines+=['','## 固定样例','', '![固定样例](examples_followup.png)','',
       '固定先导清单前3张图片，不按结果挑选。全局线性映射、范围截断与最近邻放大，没有局部修改。最后两列是不同初始噪声，不能把差异本身叫做合理语义多样性。','',
       '## 限制与可支持的判断','',
       '- 一个训练种子、Pet 64x64、四个固定噪声；既有验证图片已被前轮观察。尚无最终测试、跨数据集确认或真实后验校准证据。',
       '- 延长阶段沿用最后学习率4e-5、原模型/EMA/优化器状态，采用新记录的RNG流；不能称为原轨迹的无缝恢复，亦未改网络和损失。',
       '- LPIPS与energy score如果给出不同排序，必须并列解释，不应选择有利指标宣称全面更好。',
       '- 原图恢复、感知距离、多样性、数值收敛是不同目标。失真—感知权衡已是已知现象，不能把复现本身当成新方法。',
       '- 是否推进几何自适应预算，需要在明确目标后、只用开发集设置阈值，再在新保留图片和另一训练种子上验证。目前尚未训练该方法。','',
       '## 原始来源与证据','',
       '- [The Perception-Distortion Tradeoff](https://arxiv.org/abs/1711.06077)。',
       '- [LPIPS 官方实现](https://github.com/richzhang/PerceptualSimilarity)。',
       '- [Gneiting & Raftery 2007: Strictly Proper Scoring Rules](https://www.eecs.harvard.edu/cs286r/courses/fall10/papers/Gneiting07.pdf)。',
       '- [From Posterior Sampling to Meaningful Diversity](https://arxiv.org/abs/2310.16047)。',
       '- 本目录 summary.json / metric_source_data.csv / 各模型逐图JSON / reference_step*.json / verification.json。',
       '- 上级 FOLLOWUP_PLAN_zh.md、followup_protocol.json、NOISE_CONTROL_AMENDMENT_zh.md。',
       '- 首轮 pilot_seed0_v1 所有结果保留；本轮旧依赖噪声对照也保留，但被明确排除于比较。','']
    (RUN/'RESULTS_zh.md').write_text('\n'.join(lines))
    (RUN/'figure_qa.md').write_text('Python/matplotlib. Metrics: metric_source_data.csv; iid-noise control only.\nOne training seed, four fixed noise draws per image, 256 exploratory validation images.\nExamples are fixed first three images; uniform linear display conversion, clipping and nearest-neighbor enlargement.\nNo best-of-four selection; 128-NFE output is only a numerical reference.\n')
    print('\n'.join(table));print(json.dumps({'comparisons':comparisons,'interactions':interactions,'decomposition':decomp},indent=2))


if __name__=='__main__':main()
