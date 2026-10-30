"""Frozen-NFE paired comparison of three capacities; no test-set access."""
import argparse,csv,html,json,time
from pathlib import Path
import numpy as np
import torch
from pilot import UNet,predict,save_json,digest
from capacity_models import WideUNet
from capacity15_models import LargeUNet
from compare_capacity import interval

def read(p):return json.loads(Path(p).read_text())
def run(root):
    out=root/'runs/capacity15_30k_v6';cfg=read(out/'protocol.json')
    paths={'1.75M':root/'runs'/cfg['baseline_run'],'6.73M':root/'runs'/cfg['comparison_run'],'14.99M':out}
    records={};key=lambda r:(r['geometry'],r['nfe'],r['image_index'],r['mask_pair'])
    for label,path in paths.items():
        assert read(path/'verification.json')['passed']
        assert read(path/'seed0/summary.json')['status']=='complete'
        rows=read(path/'seed0/fm_step30000_final_rows.json');records[label]={key(r):r for r in rows}
        assert len(records[label])==5952
    assert records['1.75M'].keys()==records['6.73M'].keys()==records['14.99M'].keys()
    n=len({k[2]for k in records['14.99M']});assert n==124
    comparisons=[];source=[]
    for baseline in ['6.73M','1.75M']:
        aa=records[baseline];bb=records['14.99M']
        for nfe in cfg['nfe']:
            for geometry in ['large','fragmented','overall']:
                geoms=['large','fragmented']if geometry=='overall'else[geometry]
                for metric,sign in [('lpips_truth',1),('sample_psnr',-1),('energy_score',1)]:
                    av=np.array([np.mean([aa[g,nfe,i,j][metric]for g in geoms for j in range(4)])for i in range(n)])
                    bv=np.array([np.mean([bb[g,nfe,i,j][metric]for g in geoms for j in range(4)])for i in range(n)])
                    gains=sign*(av-bv)
                    comparisons.append({'baseline':baseline,'candidate':'14.99M','geometry':geometry,'nfe':nfe,'metric':metric,'baseline_mean':float(av.mean()),'candidate_mean':float(bv.mean()),**interval(gains)})
                    source.extend({'baseline':baseline,'candidate':'14.99M','geometry':geometry,'nfe':nfe,'metric':metric,'image_index':i,'baseline_value':float(av[i]),'candidate_value':float(bv[i]),'improvement':float(gains[i])}for i in range(n))
    primary=[r for r in comparisons if r['baseline']=='6.73M'and r['nfe']==8 and r['metric']in ['lpips_truth','sample_psnr']]
    overall=[r for r in primary if r['geometry']=='overall']
    decision={'fixed_nfe':8,'fixed_step':30000,'seed':0,'primary_baseline':'6.73M','candidate':'14.99M','minimum_effect_threshold':None,'joint_observed_improvement':all(r['mean']>0 for r in overall),'both_improvement_ci_lower_bounds_positive':all(r['ci95'][0]>0 for r in overall),'primary':primary,'interpretation':'Single training seed, reused development images; image-bootstrap intervals do not measure training-seed variation. No automatic additional training.'}
    save_json(out/'capacity_comparison.json',comparisons);save_json(out/'capacity_decision.json',decision)
    with (out/'paired_image_source_data.csv').open('w')as f:
        w=csv.DictWriter(f,fieldnames=list(source[0]));w.writeheader();w.writerows(source)
    # Remeasure all capacities in the same session, without quality metrics or I/O in timing.
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    supported=read(root/'matched_masks_v4/supported_indices.json');cache=torch.load(root/'prepared.pt',weights_only=True);bank=torch.load(root/'matched_masks_v4/mask_bank.pt',weights_only=True)
    x=cache['dev'][supported['dev'][:64]].float().cuda()/127.5-1;m=bank['dev'][:64,0,0,None].float().cuda();eps=torch.randn(x.shape,generator=torch.Generator().manual_seed(202610223)).cuda()
    latency=[]
    for label,constructor in [('1.75M',UNet),('6.73M',WideUNet),('14.99M',LargeUNet)]:
        cp=torch.load(paths[label]/'seed0/fm_step30000.pt',weights_only=False,map_location='cpu');net=constructor().cuda().eval();net.load_state_dict(cp['ema']);del cp
        for batch in [1,64]:
            for nfe in [1,4,8,16]:
                for _ in range(5):predict(net,'fm',x[:batch],m[:batch],eps[:batch],nfe)
                times=[]
                for _ in range(20):
                    torch.cuda.synchronize();start=time.perf_counter();predict(net,'fm',x[:batch],m[:batch],eps[:batch],nfe);torch.cuda.synchronize();times.append(time.perf_counter()-start)
                latency.append({'model_size':label,'batch_size':batch,'nfe':nfe,'batch_median_ms':float(np.median(times)*1000),'batch_p95_ms':float(np.percentile(times,95)*1000),'amortized_median_ms_per_image':float(np.median(times)*1000/batch),'warmups':5,'repetitions':20})
        del net
    save_json(out/'latency.json',{'rows':latency,'protocol':'Same GPU BF16 synchronized predict including known-pixel checks; GPU inputs; no I/O or LPIPS; one noise sample/image; pair0 large masks.'})
    def table(rows,fields):
        return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>'for k in fields)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>'for k in fields)+'</tr>'for r in rows)+'</table>'
    quality=[]
    for label in paths:
        for geometry in ['large','fragmented','overall']:
            geoms=['large','fragmented']if geometry=='overall'else[geometry]
            selected=[r for k,r in records[label].items()if k[0]in geoms and k[1]==8]
            quality.append({'model':label,'geometry':geometry,**{metric:float(np.mean([r[metric]for r in selected]))for metric in ['lpips_truth','sample_psnr']}})
    message='相对6.73M，14.99M的整体LPIPS下降且PSNR上升。'if decision['joint_observed_improvement']else'相对6.73M，14.99M尚未同时改善整体LPIPS与PSNR。'
    if not decision['both_improvement_ci_lower_bounds_positive']:message+='至少一个收益区间未完全高于零，请结合点估计和不确定性判断。'
    summary=read(out/'seed0/summary.json')
    doc='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>GeoStep 14.99M容量对照</title><style>body{font:15px/1.7 -apple-system,"PingFang SC",sans-serif;max-width:1250px;margin:36px auto;padding:0 22px;color:#22343e}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border-bottom:1px solid #cbd5da;text-align:left}p{max-width:1000px}</style><h1>GeoStep：继续加宽到14.99M</h1>'''
    doc+=f'<p>{message}</p><p>新模型14,986,435参数，通道96/192/288/384；seed0从头训练30k。保持原深度、时间嵌入、64×64、batch64、损失、优化器、学习率、EMA与v4匹配掩码不变。训练{summary["training_seconds"]:.1f}秒，完整训练及评估{summary["wall_seconds"]:.1f}秒。</p><h2>固定30k、NFE8：三个模型</h2>'
    doc+=table(quality,['model','geometry','lpips_truth','sample_psnr'])
    doc+='<h2>主比较：14.99M相对6.73M的增量收益</h2><p>收益为正表示改善。LPIPS用基线减候选，PSNR用候选减基线。每张原图先平均4对掩码、4个噪声，随后进行10,000次配对原图bootstrap。overall对两种几何等权平均；PSNR针对缺失区，LPIPS针对完整合成图。</p>'
    fields=['baseline','geometry','nfe','metric','baseline_mean','candidate_mean','mean','ci95']
    doc+=table(primary,fields)
    doc+='<h2>同机采样延迟</h2><p>每张图生成一个样本；预热5次、重复20次。batch_median_ms为整批耗时，batch64不能当作单图延迟。</p>'+table(latency,['model_size','batch_size','nfe','batch_median_ms','batch_p95_ms'])
    doc+='<h2>完整NFE扫描及相对1.75M的次比较</h2>'+table(comparisons,fields)
    doc+='<h2>核验与范围</h2><p>核验通过后生成此页。逐200步核对训练数据与掩码序列，10k/20k/30k核对数据与噪声生成器状态。预算相同更新数不等于相同FLOPs。仅有一个训练随机种子，图像bootstrap无法代表训练种子的波动；dev反复用于探索，未运行正式测试集。保留v4共同支持子集与近似前景匹配限制；不自动追加seed1、Regression或训练步数。</p><p><a href="verification.json">核验</a> · <a href="capacity_comparison.json">全部比较</a> · <a href="capacity_decision.json">固定主判据</a> · <a href="paired_image_source_data.csv">逐图源数据</a> · <a href="latency.json">延迟</a></p></html>'
    (out/'RESULTS_zh.html').write_text(doc)
    save_json(out/'pipeline_summary.json',{'status':'complete_verified','training_updates':30000,'model_parameters':14986435,'comparison_fixed_nfe':8,'joint_observed_improvement':decision['joint_observed_improvement'],'both_gain_ci_positive':decision['both_improvement_ci_lower_bounds_positive'],'report_sha256':digest(out/'RESULTS_zh.html'),'analysis_sha256':digest(__file__)})
    print(json.dumps(decision),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
