"""Bounded dev-only selection, one replication seed, then post-test comparison."""
import argparse,csv,html,json,subprocess,sys,time
from pathlib import Path
import numpy as np
import torch,lpips
from pilot import digest,save_json,predict
from capacity15_models import LargeUNet
from regression15_eval import evaluate
from official_test_v9 import verify_saved
from followup import check_metric_math
from compare_capacity import interval
ROOT=Path('/data/GeoStep');OUT=ROOT/'runs/perceptual_regression_30k_v10'
def read(p):return json.loads(Path(p).read_text())
def stage(name,command):
    save_json(OUT/'state.json',{'status':'running','stage':name});print('STAGE',name,flush=True)
    subprocess.run(command,check=True,cwd=ROOT)
def frozen_check():
    plan=read(OUT/'frozen_plan.json')
    for name,sha in plan['sources'].items():assert digest(ROOT/name)==sha
    for name,sha in plan['old_frozen_artifacts'].items():assert digest(ROOT/name)==sha
    assert digest(OUT/'calibration.json')==plan['calibration_sha256']
def select(cfg):
    candidates=[]
    for tag,alpha in cfg['candidates'].items():
        p=OUT/tag/'seed0';assert read(p/'verification.json')['passed'];rows=read(p/'regression_step30000_final_rows.json')
        candidates.append({'alpha_tag':tag,'alpha':alpha,'overall_dev_lpips':float(np.mean([r['lpips_truth']for r in rows])),'overall_dev_psnr':float(np.mean([r['sample_psnr']for r in rows])),'dev_rows_sha256':digest(p/'regression_step30000_final_rows.json')})
    chosen=min(candidates,key=lambda r:(r['overall_dev_lpips'],-r['overall_dev_psnr'],r['alpha']))
    result={'selected_alpha_tag':chosen['alpha_tag'],'selected_alpha':chosen['alpha'],'candidates':candidates,'rule':cfg['selection'],'test_results_read_for_selection':False,'post_test_supplement':True,'new_test_inference_started_before_selection':False}
    save_json(OUT/'selection.json',result);print('SELECTED',json.dumps(result),flush=True);return chosen['alpha_tag']
def per_image(rows,geometry,metric):
    geoms=['large','fragmented']if geometry=='overall'else[geometry]
    keys=sorted({r['image_index']for r in rows});groups={i:[]for i in keys}
    for row in rows:
        if row['geometry']in geoms:groups[row['image_index']].append(row[metric])
    return np.array([np.mean(groups[i])for i in keys])
def compare(data):
    rows=[]
    for baseline in ['MSE Regression','FM8']:
        for geometry in ['large','fragmented','overall']:
            for metric in ['lpips_truth','sample_psnr','ssim_full']:
                arrays={}
                for seed in [0,1]:
                    a=per_image(data[seed,baseline],geometry,metric);b=per_image(data[seed,'Perceptual Regression'],geometry,metric);assert len(a)==len(b)
                    gain=(a-b)if metric=='lpips_truth'else(b-a);arrays[seed]=(a,b,gain)
                    rows.append({'seed':seed,'baseline':baseline,'candidate':'Perceptual Regression','geometry':geometry,'metric':metric,'baseline_mean':float(a.mean()),'candidate_mean':float(b.mean()),'candidate_gain':interval(gain)})
                a=(arrays[0][0]+arrays[1][0])/2;b=(arrays[0][1]+arrays[1][1])/2;gain=(arrays[0][2]+arrays[1][2])/2
                rows.append({'seed':'mean_of_2_seeds','baseline':baseline,'candidate':'Perceptual Regression','geometry':geometry,'metric':metric,'baseline_mean':float(a.mean()),'candidate_mean':float(b.mean()),'candidate_gain':interval(gain)})
    return rows
def evaluate_selected(cfg,tag):
    frozen_check();selection_sha=digest(OUT/'selection.json');cal=read(OUT/'calibration.json');test=OUT/'post_test_evaluation';assert not test.exists();test.mkdir()
    checkpoints=[]
    for seed in [0,1]:
        sd=OUT/tag/f'seed{seed}';assert read(sd/'verification.json')['passed'];path=sd/'regression_step30000.pt'
        expected=next(r['sha256']for r in read(sd/'checkpoint_manifest.json')if r['step']==30000);assert digest(path)==expected
        checkpoints.append({'seed':seed,'path':str(path),'sha256':expected})
    save_json(test/'checkpoint_freeze.json',{'selection_sha256':selection_sha,'calibration_sha256':digest(OUT/'calibration.json'),'selected_alpha_tag':tag,'checkpoints':checkpoints,'post_test_supplement':True,'v9_test_results_previously_observed':True})
    v9=ROOT/'runs/official_test_two_seeds_v9';assert read(v9/'verification.json')['passed'];di=read(v9/'data_identity.json')
    assert digest(v9/'prepared_test.pt')==di['prepared_test_sha256'];assert digest(v9/'image_mask_metadata.json')==di['metadata_sha256']
    cache=torch.load(v9/'prepared_test.pt',weights_only=True);images=cache['dev'];bank=cache['masks'];assert len(images)==3585
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    metric=lpips.LPIPS(net='squeeze',version='0.1').cuda().eval()
    for p in metric.parameters():p.requires_grad_(False)
    checks=check_metric_math(metric);assert checks['passed'];save_json(test/'metric_checks.json',checks);evaluate.metric=metric
    expected_weights=read(v9/'verification.json')['lpips_weights']
    assert digest('/data/geostep_cache/torch/hub/checkpoints/squeezenet1_1-b8a52dc0.pth')==expected_weights['lpips_trunk_sha256']
    assert digest('/data/geostep_packages/lpips/weights/v0.1/squeeze.pth')==expected_weights['lpips_calibration_sha256']
    data={}
    for checkpoint in checkpoints:
        seed=checkpoint['seed'];dest=test/f'seed{seed}';dest.mkdir();stage_name=f'post_test_selected_seed{seed}'
        save_json(OUT/'state.json',{'status':'running','stage':stage_name,'alpha_tag':tag})
        c=torch.load(checkpoint['path'],weights_only=False,map_location='cpu');assert c['step']==30000 and c['seed']==seed
        net=LargeUNet().cuda().eval();net.load_state_dict(c['ema']);del c
        evaluate(net,{'dev':images},{'dev':bank},dest,{'dev_pairs_per_image':4,'noise_seed':202610225,'nfe':[1]},seed,30000,True)
        data[seed,'Perceptual Regression']=verify_saved(dest,'regression',images,bank);assert digest(checkpoint['path'])==checkpoint['sha256'];del net
        data[seed,'MSE Regression']=read(v9/f'regression_seed{seed}/regression_step30000_final_rows.json')
        data[seed,'FM8']=read(v9/f'fm_seed{seed}/fm_step30000_final_rows.json')
    comparisons=compare(data);save_json(test/'comparison.json',comparisons)
    # No additional test inference or seeds after these two selected checkpoints.
    frozen_check();assert digest(OUT/'selection.json')==selection_sha
    save_json(test/'verification.json',{'passed':True,'selected_alpha_tag':tag,'training_seeds':[0,1],'test_images':3585,'same_frozen_v9_test_cache_and_masks':True,'new_test_inference_after_dev_selection_and_checkpoint_freeze':True,'known_pixels_and_independent_numpy_sample_mse_psnr_checked':True,'v9_artifacts_unchanged':True,'post_test_supplement':True,'untouched_confirmatory_test':False})
    return data,comparisons

def benchmark(cfg,tag):
    ix=read(ROOT/'matched_masks_v4/supported_indices.json')['dev'];cache=torch.load(ROOT/'prepared.pt',weights_only=True);bank=torch.load(ROOT/'matched_masks_v4/mask_bank.pt',weights_only=True)
    x=cache['dev'][ix[:64]].float().cuda()/127.5-1;m=bank['dev'][:64,0,0,None].float().cuda();eps=torch.zeros_like(x);rows=[]
    models=[('Perceptual Regression',OUT/tag/'seed0/regression_step30000.pt','regression',1),('MSE Regression',ROOT/'runs/regression15_30k_v7/seed0/regression_step30000.pt','regression',1),('FM8',ROOT/'runs/capacity15_30k_v6/seed0/fm_step30000.pt','fm',8)]
    for label,path,kind,nfe in models:
        cp=torch.load(path,weights_only=False,map_location='cpu');net=LargeUNet().cuda().eval();net.load_state_dict(cp['ema']);del cp
        for batch in [1,64]:
            for _ in range(5):predict(net,kind,x[:batch],m[:batch],eps[:batch],nfe)
            times=[]
            for _ in range(20):
                torch.cuda.synchronize();start=time.perf_counter();predict(net,kind,x[:batch],m[:batch],eps[:batch],nfe);torch.cuda.synchronize();times.append(time.perf_counter()-start)
            rows.append({'model':label,'batch_size':batch,'nfe':nfe,'batch_median_ms':float(np.median(times)*1000),'batch_p95_ms':float(np.percentile(times,95)*1000)})
        del net
    save_json(OUT/'latency.json',{'rows':rows,'protocol':'same4090 BF16 predict, 5 warmups20 synchronized repetitions; one generated sample/image; no VGG feature extractor at inference; GPU inputs no I/O or quality metrics'})
    return rows

def report(cfg,tag,data,comparisons,latency):
    def table(rows,fields):return '<table><tr>'+''.join('<th>'+html.escape(k)+'</th>'for k in fields)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>'for k in fields)+'</tr>'for r in rows)+'</table>'
    scores=[]
    for geometry in ['large','fragmented','overall']:
        for model in ['MSE Regression','Perceptual Regression','FM8']:
            scores.append({'geometry':geometry,'model':model,**{metric:float(np.mean([per_image(data[seed,model],geometry,metric).mean()for seed in [0,1]]))for metric in ['lpips_truth','sample_psnr','ssim_full']}})
    selection=read(OUT/'selection.json');cal=read(OUT/'calibration.json')
    doc='''<!doctype html><html lang="zh"><meta charset="utf-8"><title>GeoStep 感知回归补充对照</title><style>body{font:15px/1.7 -apple-system,"PingFang SC",sans-serif;max-width:1250px;margin:36px auto;padding:0 22px;color:#22343e}table{border-collapse:collapse;width:100%;font-size:13px}td,th{padding:8px;border-bottom:1px solid #ccd5da;text-align:left}</style><h1>GeoStep：MSE＋VGG感知损失的单步回归对照</h1>'''
    doc+='<p>这是查看v9正式测试结果之后设计的补充实验，不是完全未接触测试集的事前确认性验证。v9原始结果和文件保持不变。候选权重只根据seed0 dev选择，之后固定同一个权重训练seed1；没有根据后续测试调整方案。</p>'
    doc+=f'<p>训练CNN保持14,986,435参数、同初始化/数据流/掩码、batch64、AdamW、EMA和30k更新。损失为masked MSE＋alpha×scale×VGG feature L1。VGG16使用ImageNet预训练冻结特征relu1_2/relu2_2/relu3_3；额外冻结参数{cal["frozen_feature_parameters"]:,}，只用于训练。scale={cal["feature_scale"]:.8f}由固定训练图校准。两个候选alpha为0.1/1.0，选定{tag}，实际VGG损失系数{selection["selected_alpha"]*cal["feature_scale"]:.8f}。不同特征网络不等于与LPIPS评价完全独立。</p><h2>seed0开发集的有限权重选择</h2>'
    doc+=table(selection['candidates'],['alpha_tag','alpha','overall_dev_lpips','overall_dev_psnr'])
    doc+='<h2>相同v9测试共同支持子集，两个种子指标均值</h2><p>3585/3669图，每图4对匹配掩码。Regression为单步确定性输出；FM为固定8步、4个噪声的指标平均。PSNR在缺失区域，LPIPS/SSIM在完整合成图。主指标仍为LPIPS和PSNR。</p>'+table(scores,['geometry','model','lpips_truth','sample_psnr','ssim_full'])
    display=[{**r,'gain':r['candidate_gain']['mean'],'ci95':str(r['candidate_gain']['ci95'])}for r in comparisons]
    doc+='<h2>感知回归相对各基线的配对差异</h2><p>正收益有利于感知回归。每张原图先平均掩码/噪声，再做10000次图像bootstrap；mean_of_2_seeds先逐图平均两个训练种子。区间不代表训练种子总体不确定性。</p>'+table(display,['seed','baseline','geometry','metric','baseline_mean','candidate_mean','gain','ci95'])
    doc+='<h2>同机推理成本</h2><p>推理仅使用训练后的CNN，不调用VGG。整批中位数/P95毫秒，不能把batch64整批延迟当成单图延迟。</p>'+table(latency,['model','batch_size','nfe','batch_median_ms','batch_p95_ms'])
    training=[]
    for alpha,seed in [(t,0)for t in cfg['candidates']]+[(tag,1)]:
        s=read(OUT/alpha/f'seed{seed}/summary.json');training.append({'alpha_tag':alpha,'seed':seed,'training_seconds':s['training_seconds'],'wall_seconds':s['wall_seconds']})
    doc+='<h2>训练资源</h2><p>相同CNN和更新次数不等于相同训练FLOPs或墙钟时间。两个seed0候选的搜索成本同样记录。</p>'+table(training,['alpha_tag','seed','training_seconds','wall_seconds'])
    doc+='<p><a href="selection.json">权重选择</a> · <a href="calibration.json">训练集尺度校准</a> · <a href="post_test_evaluation/comparison.json">全部配对比较</a> · <a href="post_test_evaluation/verification.json">补充测试核验</a> · <a href="latency.json">延迟</a> · <a href="../official_test_two_seeds_v9/RESULTS_zh.html">保留的v9原测试报告</a></p></html>'
    (OUT/'RESULTS_zh.html').write_text(doc)
    save_json(OUT/'summary_scores.json',scores)
    save_json(OUT/'pipeline_summary.json',{'status':'complete_verified','new_optimizer_updates':90000,'selected_alpha_tag':tag,'training_seeds':[0,1],'post_test_supplement':True,'report_sha256':digest(OUT/'RESULTS_zh.html'),'v9_unchanged':True})

def main():
    try:
        frozen_check();cfg=read(OUT/'protocol.json');p=sys.executable
        for tag in cfg['candidates']:
            stage(f'{tag}_seed0_training',[p,'train_perceptual15.py','--root',str(ROOT),'--seed','0','--alpha-tag',tag])
            stage(f'{tag}_seed0_verification',[p,'verify_perceptual15.py','--root',str(ROOT),'--seed','0','--alpha-tag',tag])
        tag=select(cfg)
        stage(f'{tag}_seed1_training',[p,'train_perceptual15.py','--root',str(ROOT),'--seed','1','--alpha-tag',tag])
        stage(f'{tag}_seed1_verification',[p,'verify_perceptual15.py','--root',str(ROOT),'--seed','1','--alpha-tag',tag])
        save_json(OUT/'state.json',{'status':'running','stage':'post_test_supplement','selected_alpha_tag':tag})
        data,comparisons=evaluate_selected(cfg,tag);latency=benchmark(cfg,tag);report(cfg,tag,data,comparisons,latency)
        save_json(OUT/'state.json',read(OUT/'pipeline_summary.json'));print('COMPLETE',json.dumps(read(OUT/'pipeline_summary.json')),flush=True)
    except Exception as e:save_json(OUT/'failure.json',{'type':type(e).__name__,'message':str(e)});raise
if __name__=='__main__':main()
