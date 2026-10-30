"""Finite seed1 replication, paired dev summary, and frozen test handoff."""
import argparse,json,subprocess,sys,html
from pathlib import Path
import numpy as np
from pilot import save_json,digest
from compare_capacity import interval
from compare_regression15 import paired

ROOT=Path('/data/GeoStep')
PAIR=ROOT/'runs/replication15_two_seeds_v8'
RUNS={0:{'fm':'capacity15_30k_v6','regression':'regression15_30k_v7'},1:{'fm':'capacity15_seed1_30k_v8','regression':'regression15_seed1_30k_v8'}}
def read(p):return json.loads(Path(p).read_text())
def stage(name,cmd):
    save_json(PAIR/'state.json',{'status':'running','stage':name})
    print('STAGE',name,flush=True);subprocess.run(cmd,check=True,cwd=ROOT)
def render(rows,title):
    fields=['seed','geometry','metric','regression_mean','fm_mean','fm_gain','ci95']
    return '<!doctype html><html lang="zh"><meta charset="utf-8"><title>'+title+'</title><style>body{font:15px/1.7 -apple-system,"PingFang SC",sans-serif;max-width:1100px;margin:40px auto;padding:0 20px}table{border-collapse:collapse;width:100%}td,th{padding:9px;border-bottom:1px solid #ccd}</style><h1>'+title+'</h1><p>14.99M，30k EMA，Regression1 vs FM8；LPIPS完整合成图，PSNR缺失区。正收益有利于FM。每图先平均4对掩码与FM的4个固定噪声；overall对两种几何等权。区间为配对原图bootstrap，不能代表训练种子总体不确定性。</p><table><tr>'+''.join('<th>'+v+'</th>'for v in fields)+'</tr>'+''.join('<tr>'+''.join('<td>'+html.escape(str(r[k]))+'</td>'for k in fields)+'</tr>'for r in rows)+'</table><p>mean_of_2_seeds先对每张图平均两个种子的指标再进行图像bootstrap。两种子不足以稳定估计训练种子方差。dev已经用于探索；正式测试结果单独报告。</p></html>'
def summarize_dev():
    rows=[];gains={}
    for seed,runs in RUNS.items():
        for kind,run in runs.items():assert read(ROOT/'runs'/run/'verification.json')['passed']
        reg=read(ROOT/'runs'/runs['regression']/f'seed{seed}/regression_step30000_final_rows.json')
        fm=read(ROOT/'runs'/runs['fm']/f'seed{seed}/fm_step30000_final_rows.json')
        for geom in ['large','fragmented','overall']:
            for met in ['lpips_truth','sample_psnr']:
                row,source=paired(reg,fm,8,geom,met)
                gain=row['fm_improvement'];rows.append({'seed':seed,'geometry':geom,'metric':met,'regression_mean':row['regression_mean'],'fm_mean':row['fm_mean'],'fm_gain':gain['mean'],'ci95':gain['ci95']})
                gains[seed,geom,met]=source
    for geom in ['large','fragmented','overall']:
        for met in ['lpips_truth','sample_psnr']:
            a=gains[0,geom,met];b=gains[1,geom,met];assert [r['image_index']for r in a]==[r['image_index']for r in b]
            values=np.array([(x['fm_improvement']+y['fm_improvement'])/2 for x,y in zip(a,b)]);inter=interval(values)
            rows.append({'seed':'mean_of_2_seeds','geometry':geom,'metric':met,'regression_mean':float(np.mean([(x['regression']+y['regression'])/2 for x,y in zip(a,b)])),'fm_mean':float(np.mean([(x['fm']+y['fm'])/2 for x,y in zip(a,b)])),'fm_gain':inter['mean'],'ci95':inter['ci95']})
    save_json(PAIR/'dev_comparison.json',rows)
    (PAIR/'RESULTS_zh.html').write_text(render(rows,'GeoStep：14.99M两种子dev复现'))
    consistent=all(r['fm_gain']>0 for r in rows if r['seed']in [0,1]and r['geometry']=='overall')
    save_json(PAIR/'dev_summary.json',{'status':'complete_verified','training_seeds':[0,1],'new_updates_this_run':60000,'both_seeds_joint_observed_fm_gain':consistent,'report_sha256':digest(PAIR/'RESULTS_zh.html'),'formal_test_run':'official_test_two_seeds_v9','training_seed_population_ci_available':False})
def freeze_test():
    # Freeze all four final checkpoint hashes BEFORE the first test-image access.
    checkpoints=[]
    for seed,runs in RUNS.items():
        for kind,run in runs.items():
            parent=ROOT/'runs'/run;assert read(parent/'verification.json')['passed']
            path=parent/f'seed{seed}/{kind}_step30000.pt'
            manifest=read(parent/'checkpoint_manifest.json');expected=next(r['sha256']for r in manifest if r['seed']==seed and r['step']==30000)
            assert digest(path)==expected
            checkpoints.append({'seed':seed,'kind':kind,'path':str(path),'sha256':expected})
    plan=read(PAIR/'frozen_plan.json')
    for name,sha in plan['sources'].items():assert digest(ROOT/name)==sha
    protocol=read(ROOT/'official_test_protocol_v9.json')
    assert digest(ROOT/'official_test_protocol_v9.json')==plan['test_protocol_sha256']
    test=ROOT/'runs'/protocol['run'];assert not test.exists();test.mkdir()
    save_json(test/'protocol.json',protocol);save_json(test/'checkpoint_freeze.json',{'checkpoints':checkpoints,'test_images_opened_before_freeze':0,'development_selection':'fixed30k EMA for both kinds and both seeds, no seed selection','source_freeze':plan,'dev_summary_sha256':digest(PAIR/'dev_summary.json')})
    save_json(PAIR/'state.json',{'status':'running','stage':'official_test','test_run':protocol['run']})
def main():
    try:
        plan=read(PAIR/'frozen_plan.json')
        for name,sha in plan['sources'].items():assert digest(ROOT/name)==sha
        p=sys.executable
        stage('fm_seed1_training',[p,'train_capacity15_seed1.py','--root',str(ROOT),'--seed','1'])
        stage('fm_seed1_verification',[p,'verify_capacity15_seed1.py','--root',str(ROOT)])
        fm=ROOT/'runs/capacity15_seed1_30k_v8'
        save_json(fm/'pipeline_summary.json',{'status':'complete_verified','training_updates':30000,'model_parameters':14986435,'verification_sha256':digest(fm/'verification.json')})
        stage('regression_seed1_preflight',[p,'preflight_regression15_seed1.py'])
        pre=read(ROOT/'runs/regression15_seed1_30k_v8/preflight.json');assert pre['passed']and pre['optimizer_updates']==0
        for name,sha in pre['source_sha256'].items():assert digest(ROOT/name)==sha
        stage('regression_seed1_training',[p,'train_regression15_seed1.py','--root',str(ROOT),'--seed','1'])
        stage('regression_seed1_verification',[p,'verify_regression15_seed1.py','--root',str(ROOT)])
        stage('seed1_dev_comparison',[p,'compare_regression15_seed1.py','--root',str(ROOT)])
        summarize_dev();freeze_test()
        stage('official_test',[p,'official_test_v9.py'])
        test=ROOT/'runs/official_test_two_seeds_v9';assert read(test/'verification.json')['passed']
        summary={'status':'complete_verified','new_training_updates':60000,'dev_report_sha256':digest(PAIR/'RESULTS_zh.html'),'test_report_sha256':digest(test/'RESULTS_zh.html'),'formal_test_run':test.name}
        save_json(PAIR/'pipeline_summary.json',summary);save_json(PAIR/'state.json',summary)
        print('ALL_COMPLETE',json.dumps(summary),flush=True)
    except Exception as e:
        save_json(PAIR/'failure.json',{'type':type(e).__name__,'message':str(e)});raise
if __name__=='__main__':main()
