"""Fixed 30k Regression1 versus FM8; paired dev metrics, secondary NFE curves."""
import argparse,csv,html,json,time
from pathlib import Path
import numpy as np
import torch
from pilot import predict,save_json,digest
from capacity15_models import LargeUNet
from compare_capacity import interval

def read(p):return json.loads(Path(p).read_text())
def paired(reg,fm,nfe,geometry,metric):
    geoms=['large','fragmented']if geometry=='overall'else[geometry]
    aa={(r['geometry'],r['image_index'],r['mask_pair']):r for r in reg}
    bb={(r['geometry'],r['image_index'],r['mask_pair']):r for r in fm if r['nfe']==nfe}
    assert aa.keys()==bb.keys()
    images=sorted({k[1]for k in aa});pairs=sorted({k[2]for k in aa})
    a=np.array([np.mean([aa[g,i,j][metric]for g in geoms for j in pairs])for i in images])
    b=np.array([np.mean([bb[g,i,j][metric]for g in geoms for j in pairs])for i in images])
    gain=(a-b)if metric=='lpips_truth'else(b-a)
    row={'geometry':geometry,'metric':metric,'fm_nfe':nfe,'regression_nfe':1,'regression_mean':float(a.mean()),'fm_mean':float(b.mean()),'fm_improvement':interval(gain)}
    source=[{'geometry':geometry,'metric':metric,'fm_nfe':nfe,'image_index':i,'regression':float(av),'fm':float(bv),'fm_improvement':float(gv)}for i,av,bv,gv in zip(images,a,b,gain)]
    return row,source

def main(root):
    out=root/'runs/regression15_seed1_30k_v8';cfg=read(out/'protocol.json');fmroot=root/'runs'/cfg['fm_run'];regroot=out/'seed1';fmdir=fmroot/'seed1'
    assert read(out/'verification.json')['passed']and read(fmroot/'verification.json')['passed']
    for rel,sha in read(out/'fm_frozen_artifacts.json').items():assert digest(fmroot/rel)==sha
    reg=read(regroot/'regression_step30000_final_rows.json');fm=read(fmdir/'fm_step30000_final_rows.json');assert len(reg)==992 and len(fm)==5952
    comparisons=[];source=[]
    for nfe in [1,2,4,8,16,32]:
        for geom in ['large','fragmented','overall']:
            for metric in ['lpips_truth','sample_psnr']:
                row,ss=paired(reg,fm,nfe,geom,metric);comparisons.append(row);source+=ss
    primary=[r for r in comparisons if r['fm_nfe']==8];overall=[r for r in primary if r['geometry']=='overall']
    decision={'fixed_updates_each':30000,'network_parameters_each':14986435,'seed':1,'primary_regression_nfe':1,'primary_fm_nfe':8,'minimum_effect_threshold':None,'fm_joint_observed_improvement':all(r['fm_improvement']['mean']>0 for r in overall),'fm_both_improvement_ci_positive':all(r['fm_improvement']['ci95'][0]>0 for r in overall),'primary':primary,'limits':cfg['limitations']}
    save_json(out/'comparison.json',comparisons);save_json(out/'decision.json',decision)
    with (out/'paired_image_source_data.csv').open('w')as f:
        w=csv.DictWriter(f,fieldnames=list(source[0]));w.writeheader();w.writerows(source)
    # Fixed pair0 learning curves, matching the FM checkpoint evaluation conditions.
    learning=[]
    for step in cfg['checkpoints']:
        rr=read(regroot/f'regression_step{step}_curve_rows.json');ff=read(fmdir/f'fm_step{step}_curve_rows.json')
        for geom in ['large','fragmented','overall']:
            for metric in ['lpips_truth','sample_psnr']:
                row,_=paired(rr,ff,8,geom,metric);learning.append({'step':step,**row})
    save_json(out/'checkpoint_comparison.json',learning)
    # Measure actual inference cost, without I/O or metric networks inside timing.
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    ix=read(root/'matched_masks_v4/supported_indices.json')['dev']
    cache=torch.load(root/'prepared.pt',weights_only=True);bank=torch.load(root/'matched_masks_v4/mask_bank.pt',weights_only=True)
    x=cache['dev'][ix[:64]].float().cuda()/127.5-1;m=bank['dev'][:64,0,0,None].float().cuda();eps=torch.randn(x.shape,generator=torch.Generator().manual_seed(202610223)).cuda()
    latency=[]
    for kind,path,filename,nfes in [('regression',regroot,'regression_step30000.pt',[1]),('fm',fmdir,'fm_step30000.pt',[1,8])]:
        cp=torch.load(path/filename,weights_only=False,map_location='cpu');net=LargeUNet().cuda().eval();net.load_state_dict(cp['ema']);del cp
        for batch in [1,64]:
            for nfe in nfes:
                for _ in range(5):predict(net,kind,x[:batch],m[:batch],eps[:batch],nfe)
                times=[]
                for _ in range(20):
                    torch.cuda.synchronize();start=time.perf_counter();predict(net,kind,x[:batch],m[:batch],eps[:batch],nfe);torch.cuda.synchronize();times.append(time.perf_counter()-start)
                latency.append({'model':kind,'batch_size':batch,'nfe':nfe,'batch_median_ms':float(np.median(times)*1000),'batch_p95_ms':float(np.percentile(times,95)*1000),'warmups':5,'repetitions':20})
        del net
    save_json(out/'latency.json',{'rows':latency,'protocol':'sameGPU BF16 synchronized predict; GPU inputs; one sample/image; includes known-pixel checks, excludes I/O and quality metrics'})
    def table(rows,fields):
        return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>'for k in fields)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>'for k in fields)+'</tr>'for r in rows)+'</table>'
    display=[]
    for row in comparisons:
        gain=row['fm_improvement'];display.append({**row,'gain':f"{gain['mean']:+.6f}",'ci95':str(gain['ci95'])})
    fields=['geometry','metric','fm_nfe','regression_mean','fm_mean','gain','ci95']
    message='固定FM8下，FM整体LPIPS和PSNR均优于同容量Regression。'if decision['fm_joint_observed_improvement']else'固定FM8下，FM未同时改善整体LPIPS和PSNR；需逐指标判断取舍。'
    if not decision['fm_both_improvement_ci_positive']:message+='至少一项收益的95%区间未完全高于零。'
    doc='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>GeoStep 14.99M Regression对照</title><style>body{font:15px/1.7 -apple-system,"PingFang SC",sans-serif;max-width:1200px;margin:36px auto;padding:0 22px;color:#22343e}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ccd5da;text-align:left}</style><h1>同容量、同掩码：Regression与Flow Matching</h1>'''
    doc+=f'<p>{message}</p><p>两者均为14,986,435参数，从完全相同的seed1初始权重开始，使用相同2812张训练图、16对/图冻结匹配掩码、图片/掩码/翻转序列、batch64、AdamW、学习率、梯度裁剪、EMA与30k更新预算。Regression在t=0直接预测图像，FM学习速度场。训练loss定义不同，不能直接比较大小。</p><h2>固定主比较：Regression1与FM8</h2><p>相同124张dev图，每图4对掩码。Regression每条件一个确定性输出；FM沿用每条件4个固定噪声样本的指标均值，不做挑样本。LPIPS针对完整合成图；PSNR针对缺失区域，使用逐样本PSNR均值。每张原图先平均掩码与噪声，再做10,000次配对图像bootstrap；正收益表示有利于FM。</p>'
    doc+=table([r for r in display if r['fm_nfe']==8],fields)
    doc+='<h2>采样成本</h2><p>两者训练更新数一致，推理成本不同。表中为整批延迟，每张图生成一个样本；不含LPIPS计算或磁盘加载。</p>'+table(latency,['model','batch_size','nfe','batch_median_ms','batch_p95_ms'])
    doc+='<h2>完整FM步数扫描</h2><p>固定主比较保持FM8，其他NFE为次要描述。</p>'+table(display,fields)
    learning_display=[{**r,'gain':r['fm_improvement']['mean'],'ci95':str(r['fm_improvement']['ci95'])}for r in learning]
    doc+='<h2>10k / 20k / 30k固定pair0验证曲线数据</h2><p>本表每图仅pair0，不能与最终4对掩码平均直接混为一组。</p>'+table(learning_display,['step']+fields)
    doc+='<h2>证据范围</h2><p>先通过核验再生成此页。训练数据随机流重放30k步，逐检查点核对数据生成器状态与FM一致，初始权重逐张量相同，最终损失进行NumPy独立复核。仍只有一个训练种子，dev已经反复用于探索；图像bootstrap不覆盖训练种子波动。继承v4共同支持子集及近似前景匹配限制。本轮没有访问正式测试集，也不自动续训或添加种子。</p><p><a href="verification.json">核验</a> · <a href="decision.json">主判据</a> · <a href="comparison.json">全部比较</a> · <a href="checkpoint_comparison.json">验证曲线</a> · <a href="paired_image_source_data.csv">逐图源数据</a> · <a href="seed1/regression_step30000_loss_summary.json">训练与验证loss</a></p></html>'
    (out/'RESULTS_zh.html').write_text(doc)
    save_json(out/'pipeline_summary.json',{'status':'complete_verified','training_updates':30000,'model_parameters':14986435,'fm_parameters':14986435,'comparison_fixed_nfe':8,'regression_nfe':1,'joint_observed_improvement':decision['fm_joint_observed_improvement'],'both_gain_ci_positive':decision['fm_both_improvement_ci_positive'],'report_sha256':digest(out/'RESULTS_zh.html'),'analysis_sha256':digest(__file__)})
    print(json.dumps(decision),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);main(p.parse_args().root)
