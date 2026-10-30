"""Correct independent-noise control; no optimizer updates."""
import argparse
import json
from pathlib import Path
import lpips
import numpy as np
import torch
from followup import aggregate, load_net, noise_bank, scores
from pilot import digest, evaluation_masks, predict, save_json


@torch.no_grad()
def run(root):
    cfg=json.loads((root/'followup_protocol.json').read_text());out=root/'runs/followup_seed0_v2'
    assert json.loads((out/'summary.json').read_text())['status']=='complete'
    torch.set_num_threads(4)
    images=torch.load(root/'prepared.pt',weights_only=True)['eval'].float()/127.5-1
    ms=evaluation_masks(len(images),'eval');noise=noise_bank(images.shape,cfg['noise_seed']+1)
    metric=lpips.LPIPS(net='squeeze',version='0.1').cuda().eval()
    for step in [2000,10000]:
        path=out/f'iid_noise_step{step}_eval_rows.json'
        if path.exists():raise RuntimeError('Refusing to overwrite corrected noise evaluation')
        regpath=(root/'runs/pilot_seed0_v1' if step==2000 else out)/f'regression_step{step}.pt'
        net=load_net(regpath)
        fmrows=json.loads((out/f'fm_step{step}_eval_rows.json').read_text())
        allrows=[];examples={}
        for geom,mask in ms.items():
            vv=sorted([r for r in fmrows if r['geometry']==geom and r['nfe']==16],key=lambda r:r['index'])
            # Existing ensemble_variance uses image units [0,1]. Multiply by 4
            # to move to [-1,1], then by S/(S-1) to match expected finite variance.
            target=torch.tensor([r['ensemble_variance'] for r in vv],device='cuda')
            for begin in range(0,len(images),32):
                end=min(begin+32,len(images));x=images[begin:end].cuda();m=mask[begin:end].cuda()
                p=predict(net,'regression',x,m,noise[0,begin:end].cuda()).clamp(-1,1)
                sigma=(4*(4/3)*target[begin:end]).sqrt()
                raw=p[None]+m[None]*noise[:,begin:end].cuda()*sigma[None,:,None,None,None]
                pred=raw.clamp(-1,1);values=scores(pred,raw,x,m,metric)
                for j in range(len(x)):
                    allrows.append({'index':begin+j,'geometry':geom,'model':'regression_iid_noise',
                                    'step':step,'nfe':1,'expected_raw_finite_variance':float(target[begin+j]),
                                    **{k:float(v[j]) for k,v in values.items()}})
                if begin==0:examples[geom]=pred[:,:8].cpu().numpy()
        save_json(path,allrows);save_json(out/f'iid_noise_step{step}_eval_summary.json',aggregate(allrows))
        np.savez_compressed(out/f'iid_noise_step{step}_eval_examples.npz',**examples)
        print('CORRECT_IID_NOISE',step,json.dumps(aggregate(allrows)),flush=True)
    save_json(out/'noise_amendment_identity.json',{'source_sha256':digest(__file__),
       'amendment_sha256':digest(root/'NOISE_CONTROL_AMENDMENT_zh.md'),
       'additional_optimizer_updates':0,'original_control_excluded_from_final_comparisons':True,
       'independent_gaussian_samples':True})
    save_json(out/'evaluation_catalog.json',{
       'approved_eval_row_files':[f'{kind}_step{step}_eval_rows.json'
          for step in [2000,10000] for kind in ['regression','fm','iid_noise']],
       'excluded_control_row_files':[f'noise_control_step{step}_eval_rows.json' for step in [2000,10000]],
       'exclusion_reason':'Original noise ensembles were centered and normalized jointly; samples not independent for energy U-statistic interpretation.',
       'summary_json_note':'Original summary.json preserves historical metric_groups including the excluded control. Use this catalog and corrected iid-noise files for scientific comparisons.'})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    run(p.parse_args().root)
