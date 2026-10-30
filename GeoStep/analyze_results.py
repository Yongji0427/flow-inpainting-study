"""Render only saved observations; no training, tuning or example selection."""
import csv
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw

ROOT=Path(__file__).resolve().parent
RUN=ROOT/'runs/pilot_seed0_v1'


def load(name):return json.loads((RUN/name).read_text())


def main():
    s=load('summary.json');gain=load('paired_gain.json');ref=load('reference_check.json')
    assert load('verification.json')['passed']
    colors={'scattered':'#247B9E','large':'#B75A42'}
    names={'scattered':'Scattered patches','large':'Large rectangle'}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'svg.fonttype':'none','pdf.fonttype':42})
    metrics=s['metrics']
    with (RUN/'metric_source_data.csv').open('w') as f:
        fields=['model','geometry','nfe','mse','raw_mse','psnr','ssim_full','out_of_range_fraction','images','evaluation_seconds']
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(metrics)
    rows=load('fm_per_image.json')
    hist_data={}
    for geom in colors:
        lo=sorted((r for r in rows if r['geometry']==geom and r['nfe']==1),key=lambda r:r['index'])
        hi=sorted((r for r in rows if r['geometry']==geom and r['nfe']==16),key=lambda r:r['index'])
        hist_data[geom]=np.array([b['psnr']-a['psnr'] for a,b in zip(lo,hi)])
    all_deltas=np.concatenate(list(hist_data.values()))
    common_bins=np.linspace(all_deltas.min(),all_deltas.max(),26)
    fig,ax=plt.subplots(2,2,figsize=(11,8.3),layout='constrained')
    for geom in colors:
        data=sorted((r for r in metrics if r['model']=='fm' and r['geometry']==geom),key=lambda r:r['nfe'])
        reg=next(r for r in metrics if r['model']=='regression' and r['geometry']==geom)
        for axis,key,label in [(ax[0,0],'psnr','Missing-region PSNR (dB)'),(ax[0,1],'ssim_full','Composite-image SSIM')]:
            axis.plot([r['nfe'] for r in data],[r[key] for r in data],'-o',color=colors[geom],label=names[geom]+' / FM')
            axis.axhline(reg[key],ls='--',color=colors[geom],alpha=.75,label=names[geom]+' / regression')
            axis.set_xscale('log',base=2);axis.set_xticks([1,2,4,8,16,32],[1,2,4,8,16,32]);axis.set_xlabel('Network evaluations (Euler)');axis.set_ylabel(label)
        a=sorted((r for r in rows if r['geometry']==geom and r['nfe']==1),key=lambda r:r['index'])
        b=sorted((r for r in rows if r['geometry']==geom and r['nfe']==16),key=lambda r:r['index'])
        delta=np.array([y['psnr']-x['psnr'] for x,y in zip(a,b)])
        ax[1,0].hist(delta,bins=common_bins,histtype='step',lw=1.8,color=colors[geom],label=names[geom])
        data=sorted((r for r in ref if r['geometry']==geom and r['solver']=='euler'),key=lambda r:r['nfe'])
        ax[1,1].plot([r['nfe'] for r in data],[r['mse_to_64nfe_reference'] for r in data],'-o',color=colors[geom],label=names[geom])
    ax[0,0].legend(fontsize=8)
    ax[1,0].axvline(0,color='#777777',lw=.8);ax[1,0].set_xlabel('PSNR gain: 16 NFE minus 1 NFE (dB)');ax[1,0].set_ylabel('Images');ax[1,0].legend(fontsize=8)
    ax[1,1].set_xscale('log',base=2);ax[1,1].set_yscale('log');ax[1,1].set_xticks([1,4,16,32],[1,4,16,32]);ax[1,1].set_xlabel('Network evaluations (Euler)');ax[1,1].set_ylabel('MSE to 64-NFE output');ax[1,1].legend(fontsize=8)
    for axis,label in zip(ax.flat,['a','b','c','d']):axis.text(-.12,1.04,label,transform=axis.transAxes,fontweight='bold',fontsize=12)
    fig.suptitle('GeoStep exploratory pilot | one seed, Pet 64 x 64, equal missing area',fontsize=13)
    fig.savefig(RUN/'pilot_results.png',dpi=180)
    fig.savefig(RUN/'pilot_results.svg');fig.savefig(RUN/'pilot_results.pdf')
    plt.close(fig)
    fig,axes=plt.subplots(1,2,figsize=(10,3.2),layout='constrained')
    for axis,kind in zip(axes,['regression','fm']):
        hist=[json.loads(x) for x in (RUN/f'{kind}_train.jsonl').read_text().splitlines()]
        axis.plot([r['step'] for r in hist],[r['loss_mean_last100'] for r in hist],color='#247B9E')
        axis.set_xlabel('Optimizer updates');axis.set_ylabel('Training loss (last 100 updates)');axis.set_title(kind.capitalize()+' objective')
    fig.suptitle('Different objectives: loss values must not be compared across panels',fontsize=10)
    fig.savefig(RUN/'training_curves.png',dpi=180);fig.savefig(RUN/'training_curves.svg');plt.close(fig)
    # Re-layout fixed first three examples, preserving the pixels in the full plate.
    plate=Image.open(RUN/'examples.png').convert('RGB')
    compact=Image.new('RGB',(640,34+6*112+25),'white')
    draw=ImageDraw.Draw(compact)
    for j,label in enumerate(['Original','Masked','Regression','FM 1','FM 16']):
        draw.text((84+j*112,8),label,fill='black')
    for row,source_row in enumerate([0,1,2,8,9,10]):
        if row in [0,3]:draw.text((3,40+row*112),'Scattered' if row==0 else 'Large',fill='black')
        for j,source_col in enumerate([0,1,2,3,5]):
            tile=plate.crop((source_col*112,34+source_row*112,(source_col+1)*112,34+(source_row+1)*112))
            compact.paste(tile,(80+j*112,34+row*112))
    draw.text((4,34+6*112+5),'Fixed first 3 evaluation images; same images in both mask conditions.',fill='black')
    compact.save(RUN/'examples_compact.png')
    header='| Geometry | Regression | FM 1 | FM 2 | FM 4 | FM 8 | FM 16 | FM 32 |\n|---|---:|---:|---:|---:|---:|---:|---:|'
    table=[header]
    for geom in colors:
        row=[next(r for r in metrics if r['model']=='regression' and r['geometry']==geom)['psnr']]
        row += [next(r for r in metrics if r['model']=='fm' and r['geometry']==geom and r['nfe']==k)['psnr'] for k in [1,2,4,8,16,32]]
        table.append('| '+('分散小块' if geom=='scattered' else '连续大块')+' | '+' | '.join(f'{v:.3f}' for v in row)+' |')
    ci=gain['mse']['paired_image_bootstrap_95ci']
    direction='正向' if ci[0]>0 else ('反向' if ci[1]<0 else '尚不明确')
    lines=[
      '# GeoStep 有界先导实验',
      '',
      '日期：2026-10-21。GeoStep = Geometry and Sampling Steps。',
      '',
      f'本轮已完成：回归 CNN 与条件 FM CNN 各 2,000 次更新，共 4,000 次；另有 20 次准备检查更新。主进程实际耗时 {s["wall_seconds"]:.1f} 秒。独立产物复核通过。',
      '',
      f'按预先指定的逐像素误差判据，几何交互方向为 **{direction}**：本轮没有支持“大洞从增加步数中获得更多原图恢复收益”的假设。这仅指这个检查点、这个数据域与固定掩码协议，不代表多种子确认或普遍规律。',
      '',
      '## 冻结协议与真实执行边界',
      '',
      '- GPU：服务器实测 RTX 4090，49,140 MiB 总显存；模型 1,747,523 参数，batch 64，BF16。',
      '- 数据：已有 Oxford-IIIT Pet 已审阅划分；2,924 张训练图片、128 张开发图片、256 张先导评价图片。后两者来自已有 730 张验证集，互不重叠。',
      '- 未访问 3,669 张最终测试图片。已有验证集曾用于 LGFNet；本轮不将其称为全新最终测试。',
      '- 64x64 中心裁切缩放，训练仅水平翻转。两类模型同结构与初始权重，同图片/掩码/翻转采样规则；回归预测干净像素，FM 预测速度。',
      '- 两种几何均缺失 1,638/4,096 像素，约 39.99%。同一图片在两个几何下评价；各 NFE 复用同一掩码与初始噪声。',
      '- 评价固定的第 2,000 步 EMA 检查点，没有在先导评价集上挑选模型。',
      '',
      '## 观察结果',
      '',
      '下表为缺失区域的逐图 PSNR 均值（dB，越高越接近原图）。输出先截断到显示范围 [0,1]；未截断 MSE 单独保存，不能把截断收益归为采样收益。',
      '',
      '\n'.join(table),
      '',
      '从 1 增至 16 NFE 的误差下降（正值表示改善）：',
      '',
      '| 指标 | 分散小块收益 | 连续大块收益 | 大块减小块 | 配对图像 bootstrap 95% CI |',
      '|---|---:|---:|---:|---|',
    ]
    for metric,label in [('mse','截断 MSE 下降'),('raw_mse','未截断 MSE 下降'),('psnr','PSNR 提升 (dB)'),('ssim_full','全图 SSIM 提升')]:
        g=gain[metric];low,high=g['paired_image_bootstrap_95ci']
        lines.append(f'| {label} | {g["gain_scattered"]:.6f} | {g["gain_large"]:.6f} | {g["large_minus_scattered_gain"]:.6f} | [{low:.6f}, {high:.6f}] |')
    lines += [
      '',
      '区间通过对 256 张图片作配对重采样得到；每张图片保留两种几何与两种 NFE 的对应关系。它只描述固定模型下的图像变化，不包含重新训练的随机性。',
      '',
      '![结果图](pilot_results.png)',
      '',
      '图 a/b：质量随网络调用次数的变化；虚线是直接回归基线。图 c：每张图片的 1→16 NFE PSNR 增益。图 d：与相同初始噪声的 64-NFE 数值参考的偏差，仅 64 张图片；参考输出不是真值，也未确认连续时间极限。',
      '',
      '![训练曲线](training_curves.png)',
      '',
      '训练损失只用于检查各自目标的学习过程；两个目标的数值尺度不同，不能直接比较。',
      '',
      '![固定样例](examples.png)',
      '',
      '样例使用评价清单前 8 张随机固定图片，没有按表现挑选。上半为分散遮挡，下半为大块遮挡；两半图片相同。显示采用 [-1,1] 到 [0,255] 的全局线性映射与范围截断、最近邻放大；没有局部增强或生成内容修饰。',
      '',
      '## 解释边界与下一步',
      '',
      '**当前最重要的修正是把数值收敛和原图恢复分开。** 在 64 张参考图片上，从 1 增至 32 NFE，分散掩码对 64-NFE 输出的 MSE 从约 0.01151 降至 0.00001205，大块掩码从约 0.02487 降至 0.00005105；但对真实原图的误差同时增大。这说明更精细地采样当前模型，并不自动得到更接近真实原图的结果。64-NFE 只是固定数值参照，尚未证明为连续极限。',
      '',
      '**固定样例的视觉观察：** 直接回归在连续大洞内往往给出平滑、模糊的填充；FM 从少步到多步出现更多轮廓和纹理，同时仍有明显错误部件、结构不一致与残余伪影。不能仅凭这些样例确认感知质量更好，更不能声称已恢复正确语义。',
      '',
      '**为何 PSNR 不是充分的生成判据：** 对一个存在多种合理答案的条件 y，条件均值 μ(y) 在平方误差意义下最优。若真实内容 X 和一个理想模型生成的独立样本 X′ 都服从相同的条件分布，则 E[||X−μ||²|y]=tr Cov(X|y)，而 E[||X−X′||²|y]=2 tr Cov(X|y)。这是条件独立的理想化推导，不是本轮模型满足该分布假设的证据；它说明仅比较一个生成样本与一张真值，无法单独判断条件生成分布的好坏。',
      '',
      '**推进建议：** 先保留这个便宜、可运行的管线；暂不扩大训练或直接开展“几何自适应预算”。下一轮若获授权，应先确定研究目标究竟是贴近原图、生成合理细节，还是数值采样效率，并配置对应评价。涉及多解生成时，需要固定多个噪声样本、感知/分布与条件一致性指标，不能仅以 PSNR 翻转作为通过标准。',
      '',
      '- 能跑通、损失下降、指标矩阵完整，构成实现与训练可行性证据；是否值得推进要同时看基线、样例与几何交互，而非只看运行成功。',
      '- 普通 FM 的 1-NFE 是粗采样，不代表专门训练的一步生成器。没有实现 MeanFlow，不能得出一步生成能力的普遍结论。',
      '- 掩码改变了周长、位置分布和上下文；本轮未把这些因素逐一分离。距离与采样收益的关联不能直接解释为语义不确定性。',
      '- 单种子、小模型、宠物图片、64x64、一次初始噪声，结果可能受训练不足影响。没有 LPIPS/FID，也不能仅凭 MSE/SSIM 确认语义合理性或感知质量。',
      '- 没有开展自适应预算、训练扩展、多种子或正式测试。若继续，应先依据这些结果决定是否需要提高模型/训练充分性，或检验几何关系的稳定性，再冻结下一轮协议。',
      '',
      '## 证据入口',
      '',
      '- `summary.json`、`metric_source_data.csv`：完整汇总指标与实际执行计数。',
      '- `regression_per_image.json`、`fm_per_image.json`：逐图结果，包含距离分组误差、范围外比例。',
      '- `paired_gain.json`：配对增益与区间；`reference_check.json`：Euler/Heun 与高步数参考。',
      '- `verification.json`：独立算术重算、保存样例重算、身份与检查点更新计数复核。',
      '- 项目上级 `protocol.json`、`data_manifest.json`、`preflight.json`：冻结设置、图片 ID、文件哈希与准备检查。',
      '- `identity.json` 绑定实际训练源码、协议和数据清单；服务器项目 `/data/GeoStep`。',
    ]
    (RUN/'RESULTS_zh.md').write_text('\n'.join(lines)+'\n')
    (RUN/'figure_qa.md').write_text('Backend: Python/matplotlib. Source: metric_source_data.csv, per-image JSON, reference_check.json.\nOne training seed; paired image bootstrap intervals do not represent training variation.\nExamples: first 8 images in the prespecified shuffled pilot evaluation manifest; no outcome-based selection.\nImage display: common linear conversion, clipping, nearest-neighbor enlargement.\n')
    print('\n'.join(table))
    print(json.dumps(gain,indent=2))


if __name__=='__main__':main()
