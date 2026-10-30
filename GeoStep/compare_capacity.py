"""Paired width-control metrics and measured latency; no training or threshold tuning."""
import argparse,csv,html,json,time
from pathlib import Path
import numpy as np
import torch
from pilot import UNet,autocast,predict,save_json,digest
from capacity_models import WideUNet

def read(p):return json.loads(Path(p).read_text())
def interval(values):
    values=np.asarray(values,dtype=float);rng=np.random.default_rng(202610222)
    boot=values[rng.integers(len(values),size=(10000,len(values)))].mean(1)
    return {'mean':float(values.mean()),'ci95':np.percentile(boot,[2.5,97.5]).tolist(),'images':len(values)}

def benchmark(root,out,cfg):
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    supported=read(root/'matched_masks_v4/supported_indices.json')
    cache=torch.load(root/'prepared.pt',weights_only=True)
    bank=torch.load(root/'matched_masks_v4/mask_bank.pt',weights_only=True)
    x=cache['dev'][supported['dev'][:64]].float().cuda()/127.5-1;m=bank['dev'][:64,0,0,None].float().cuda()
    eps=torch.randn(x.shape,generator=torch.Generator().manual_seed(202610223)).cuda()
    rows=[]
    for label,constructor,path in [('small',UNet,root/'runs'/cfg['baseline_run']/'seed0'),('wide',WideUNet,out/'seed0')]:
        c=torch.load(path/'fm_step30000.pt',weights_only=False,map_location='cpu')
        net=constructor().cuda().eval();net.load_state_dict(c['ema']);del c
        for batch in [1,64]:
            for nfe in [1,4,8,16]:
                for _ in range(5):predict(net,'fm',x[:batch],m[:batch],eps[:batch],nfe)
                times=[]
                for _ in range(20):
                    torch.cuda.synchronize();start=time.perf_counter();predict(net,'fm',x[:batch],m[:batch],eps[:batch],nfe);torch.cuda.synchronize();times.append(time.perf_counter()-start)
                rows.append({'model_size':label,'batch_size':batch,'nfe':nfe,'batch_median_ms':float(np.median(times)*1000),'batch_p95_ms':float(np.percentile(times,95)*1000),'amortized_median_ms_per_image':float(np.median(times)*1000/batch),'warmups':5,'repetitions':20,'one_noise_sample_per_image':True})
        del net
    save_json(out/'latency.json',{'rows':rows,'protocol':'Same GPU, BF16 autocast, synchronized predict() including known-pixel checks; GPU-resident inputs, no disk loading or metric network; first64 dev images, pair0 large masks; one generated sample/image. Batch latency and amortized latency are distinct.'})
    return rows

def run(root):
    out=root/'runs/wide_capacity_30k_v5';cfg=read(out/'protocol.json');base=root/'runs'/cfg['baseline_run'];wide=out/'seed0'
    assert read(out/'verification.json')['passed']and read(base/'verification.json')['passed']
    assert read(wide/'summary.json')['status']=='complete'
    a=read(base/'seed0/fm_step30000_final_rows.json');b=read(wide/'fm_step30000_final_rows.json')
    key=lambda r:(r['geometry'],r['nfe'],r['image_index'],r['mask_pair'])
    aa={key(r):r for r in a};bb={key(r):r for r in b};assert aa.keys()==bb.keys()and len(aa)==5952
    n=len(set(r['image_index']for r in a));assert n==124
    comparisons=[];source=[]
    for nfe in cfg['nfe']:
        for geometry in ['large','fragmented','overall']:
            selected=['large','fragmented']if geometry=='overall'else[geometry]
            for metric,sign in [('lpips_truth',1),('sample_psnr',-1),('energy_score',1)]:
                gains=[];smalls=[];wides=[]
                for i in range(n):
                    av=np.mean([aa[g,nfe,i,j][metric]for g in selected for j in range(4)])
                    bv=np.mean([bb[g,nfe,i,j][metric]for g in selected for j in range(4)])
                    gains.append(float(sign*(av-bv)));smalls.append(float(av));wides.append(float(bv))
                    source.append({'geometry':geometry,'nfe':nfe,'image_index':i,'metric':metric,'small':float(av),'wide':float(bv),'improvement':float(sign*(av-bv))})
                comparisons.append({'geometry':geometry,'nfe':nfe,'metric':metric,'small_mean':float(np.mean(smalls)),'wide_mean':float(np.mean(wides)),**interval(gains)})
    primary=[r for r in comparisons if r['nfe']==8 and r['metric']in ['lpips_truth','sample_psnr']]
    overall=[r for r in primary if r['geometry']=='overall']
    decision={'fixed_nfe':8,'fixed_step':30000,'seed':0,'minimum_effect_threshold':None,'joint_observed_improvement':all(r['mean']>0 for r in overall),'both_improvement_ci_lower_bounds_positive':all(r['ci95'][0]>0 for r in overall),'primary':primary,'interpretation':'Positive point estimates are observed changes, not automatically established gains. CI is paired original-image bootstrap, not a training-seed population CI. This is a single-seed exploratory comparison on reused dev. No automatic additional training.'}
    save_json(out/'capacity_comparison.json',comparisons);save_json(out/'capacity_decision.json',decision)
    with (out/'paired_image_source_data.csv').open('w')as f:
        w=csv.DictWriter(f,fieldnames=list(source[0]));w.writeheader();w.writerows(source)
    latency=benchmark(root,out,cfg)
    def fmt(r):return f"{r['mean']:+.6f} [{r['ci95'][0]:+.6f}, {r['ci95'][1]:+.6f}]"
    table=''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>'for v in [r['geometry'],r['metric'],f"{r['small_mean']:.6f}",f"{r['wide_mean']:.6f}",fmt(r)])+'</tr>'for r in primary)
    cost=''.join(f"<tr><td>{r['model_size']}</td><td>{r['batch_size']}</td><td>{r['nfe']}</td><td>{r['batch_median_ms']:.2f}</td><td>{r['batch_p95_ms']:.2f}</td></tr>"for r in latency)
    curve=''.join('<tr>'+''.join(f'<td>{html.escape(str(v))}</td>'for v in [r['geometry'],r['nfe'],r['metric'],f"{r['small_mean']:.6f}",f"{r['wide_mean']:.6f}",fmt(r)])+'</tr>'for r in comparisons if r['geometry']!='overall'and r['metric']in ['lpips_truth','sample_psnr'])
    message='NFE8下，整体LPIPS下降且PSNR上升。'if decision['joint_observed_improvement']else'NFE8下，尚未出现整体LPIPS与PSNR同时改善。'
    if decision['joint_observed_improvement']and not decision['both_improvement_ci_lower_bounds_positive']:message+='至少一个收益区间跨零，现有差异仍不确定。'
    doc=f'''<!doctype html><html lang="zh"><meta charset="utf-8"><title>GeoStep · 6.73M容量对照</title><style>body{{font:16px/1.7 -apple-system,BlinkMacSystemFont,"PingFang SC",sans-serif;max-width:1150px;margin:40px auto;padding:0 22px;color:#22343e}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:9px;border-bottom:1px solid #cbd5da;text-align:left}}.card{{background:#eef3f6;padding:18px;border-radius:12px}}a{{color:#267b9c}}</style><h1>GeoStep：纯加宽能否同时改善LPIPS和PSNR？</h1><p class="card">{message}本次仅为seed0探索；取消5%硬门槛，但仍区分观察到的变化与有区间支持的改善。</p><p>原CNN：1,747,523参数，通道32/64/96/128；加宽CNN：6,734,723参数，通道64/128/192/256。沿用原前向逻辑、128维时间嵌入、深度、BF16、损失、AdamW、EMA和30k配方；重新训练而非从小模型热启动。沿用2812张训练图、124张dev图及同一v4掩码库。逐200步核对数据/掩码序列，10k/20k/30k核对数据和噪声生成器状态。</p><h2>固定主比较：30k、NFE8</h2><p>收益统一以正数为好：LPIPS小模型−大模型；PSNR大模型−小模型。各图先平均4对掩码和4个噪声，再做10,000次配对原图bootstrap。overall为两种几何等权平均；每种几何单列，防止平均值掩盖退化。PSNR计算缺失区，LPIPS沿用完整合成图。</p><table><tr><th>几何</th><th>指标</th><th>Small</th><th>Wide</th><th>改善 [95% CI]</th></tr>{table}</table><h2>实测采样成本</h2><p>相同4090、BF16、GPU驻留输入，每张图生成一个样本；同步计时，预热5次、重复20次。包含predict的已知像素检查，不含磁盘加载和LPIPS计算。表中为整批耗时，batch64不能当作单图延迟。</p><table><tr><th>模型</th><th>Batch</th><th>NFE</th><th>整批中位数ms</th><th>整批P95 ms</th></tr>{cost}</table><h2>完整NFE扫描</h2><p>其他步数用于描述质量与计算取舍，不重新选择最优点替代固定NFE8主比较。</p><table><tr><th>几何</th><th>NFE</th><th>指标</th><th>Small</th><th>Wide</th><th>改善 [95% CI]</th></tr>{curve}</table><h2>核验与范围</h2><p>本轮核验通过才生成此页。固定更新量不等于固定FLOPs；改善不能直接归因于已证明的容量瓶颈。掩码继承v4的近似前景匹配与共同支持子集限制。现有dev反复用于探索，GeoStep本轮未访问正式测试图像。没有追加seed1或Regression训练；正式方法对照仍需同掩码、同规模基线。</p><p><a href="verification.json">核验记录</a> · <a href="capacity_decision.json">联合指标判定</a> · <a href="capacity_comparison.json">全部指标与区间</a> · <a href="latency.json">实测延迟</a> · <a href="paired_image_source_data.csv">逐图源数据</a> · <a href="../matched_geometry_30k_v4/RESULTS_zh.html">原v4结果</a></p></html>'''
    (out/'RESULTS_zh.html').write_text(doc)
    save_json(out/'pipeline_summary.json',{'status':'complete_verified','training_updates':30000,'model_parameters':6734723,'comparison_fixed_nfe':8,'joint_observed_improvement':decision['joint_observed_improvement'],'both_gain_ci_positive':decision['both_improvement_ci_lower_bounds_positive'],'report_sha256':digest(out/'RESULTS_zh.html'),'analysis_sha256':digest(__file__)})
    print(json.dumps(decision),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
