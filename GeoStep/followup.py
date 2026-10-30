"""Frozen v2 metrics and bounded warm-start extension; original run is read-only."""
import argparse
import copy
import hashlib
import itertools
import json
import os
from pathlib import Path
import time

import lpips
import numpy as np
import torch
from PIL import Image, ImageDraw
from torch import nn

from pilot import (UNet, autocast, digest, evaluation_masks, masks, predict,
                   save_json, seed_all, ssim_per_image)


def noise_bank(shape, seed):
    return torch.randn((4,)+tuple(shape),generator=torch.Generator().manual_seed(seed))


@torch.no_grad()
def scores(pred, raw, x, m, perceptual):
    # pred/raw: S,B,C,H,W, represented in [-1,1].
    samples,batch=pred.shape[:2]
    weights=m[None]
    den=3*m.sum((1,2,3))
    sq=((pred-x[None])/2).square()
    per_mse=(sq*weights).sum((2,3,4))/den[None]
    mean=pred.mean(0)
    mean_mse=((((mean-x)/2).square()*m).sum((1,2,3))/den)
    variance=((((pred-mean[None])/2).square()*weights).sum((2,3,4))/den[None]).mean(0)
    assert torch.allclose(per_mse.mean(0),mean_mse+variance,atol=2e-7,rtol=2e-6)
    pflat=pred.flatten(0,1);truth=x[None].expand(samples,-1,-1,-1,-1).reshape_as(pflat)
    truth_lpips=perceptual(pflat,truth).reshape(samples,batch).mean(0)
    ssim=ssim_per_image(pflat,truth).reshape(samples,batch).mean(0)
    pairs=list(itertools.combinations(range(samples),2))
    pa=torch.cat([pred[i] for i,j in pairs]);pb=torch.cat([pred[j] for i,j in pairs])
    pair_lpips=perceptual(pa,pb).reshape(len(pairs),batch).mean(0)
    rms_pairs=torch.stack([((((pred[i]-pred[j])/2).square()*m).sum((1,2,3))/den).sqrt() for i,j in pairs])
    energy=per_mse.sqrt().mean(0)-0.5*rms_pairs.mean(0)
    raw_mse=((((raw-x[None])/2).square()*weights).sum((2,3,4))/den[None]).mean(0)
    raw_var=((((raw-raw.mean(0)[None])/2).square()*weights).sum((2,3,4))/den[None]).mean(0)
    oor=((((raw<-1)|(raw>1))*weights).sum((2,3,4))/den[None]).mean(0)
    return {k:v.cpu().numpy() for k,v in {
        'sample_mse':per_mse.mean(0),'sample_psnr':(-10*per_mse.clamp_min(1e-12).log10()).mean(0),
        'mean_prediction_mse':mean_mse,'ensemble_variance':variance,'raw_mse':raw_mse,'raw_variance':raw_var,
        'lpips_truth':truth_lpips,'lpips_pairwise':pair_lpips,'energy_score':energy,
        'ssim_full':ssim,'out_of_range_fraction':oor}.items()}


def load_net(path):
    c=torch.load(path,weights_only=False,map_location='cpu')
    net=UNet().cuda();net.load_state_dict(c['ema']);net.eval()
    return net


@torch.no_grad()
def evaluate(root,out,net,kind,step,images,split,metric,cfg,nfes=None):
    nfes=([1] if kind=='regression' else cfg['nfe']) if nfes is None else nfes
    ms=evaluation_masks(len(images),split)
    eps=noise_bank(images.shape,cfg['noise_seed'])
    rows=[];examples={};last16={}
    for geom,mask in ms.items():
        for nfe in nfes:
            clock=time.perf_counter();chunkrows=[]
            for begin in range(0,len(images),32):
                end=min(begin+32,len(images));b=end-begin
                x=images[begin:end].cuda();m=mask[begin:end].cuda();e=eps[:,begin:end].cuda()
                if kind=='regression':
                    raw=predict(net,kind,x,m,e[0],1)[None].expand(4,-1,-1,-1,-1)
                else:
                    xr=x[None].expand(4,-1,-1,-1,-1).reshape(4*b,3,64,64)
                    mr=m[None].expand(4,-1,-1,-1,-1).reshape(4*b,1,64,64)
                    raw=predict(net,kind,xr,mr,e.reshape(4*b,3,64,64),nfe).reshape(4,b,3,64,64)
                pred=raw.clamp(-1,1)
                values=scores(pred,raw,x,m,metric)
                for j in range(b):
                    row={'index':begin+j,'geometry':geom,'model':kind,'step':step,'nfe':nfe,
                         **{k:float(v[j]) for k,v in values.items()}}
                    rows.append(row);chunkrows.append(row)
                if begin==0:examples[(geom,nfe)]=pred[:,:8].cpu()
                if nfe==16:last16.setdefault(geom,[]).append(pred.cpu())
            summary={k:float(np.mean([r[k] for r in chunkrows])) for k in values}
            print('METRICS',json.dumps({'split':split,'model':kind,'step':step,'geometry':geom,'nfe':nfe,
                      'seconds':time.perf_counter()-clock,**summary}),flush=True)
    last16={k:torch.cat(v,dim=1) for k,v in last16.items()}
    stem=f'{kind}_step{step}_{split}'
    save_json(out/(stem+'_rows.json'),rows)
    summaries=aggregate(rows)
    save_json(out/(stem+'_summary.json'),summaries)
    np.savez_compressed(out/(stem+'_examples.npz'),**{f'{g}_{n}':v.numpy() for (g,n),v in examples.items()})
    return rows,summaries,examples,last16


def aggregate(rows):
    output=[]
    keys=[k for k in rows[0] if k not in ['index','geometry','model','step','nfe']]
    groups=sorted({(r['model'],r['step'],r['geometry'],r['nfe']) for r in rows})
    for model,step,geometry,nfe in groups:
        rr=[r for r in rows if (r['model'],r['step'],r['geometry'],r['nfe'])==(model,step,geometry,nfe)]
        output.append({'model':model,'step':step,'geometry':geometry,'nfe':nfe,'images':len(rr),
                       **{k:float(np.mean([r[k] for r in rr])) for k in keys}})
    return output


@torch.no_grad()
def white_noise_control(out,reg,step,images,split,metric,cfg,fm16):
    ms=evaluation_masks(len(images),split);eps=noise_bank(images.shape,cfg['noise_seed']+1)
    rows=[]
    for geom,mask in ms.items():
        for begin in range(0,len(images),32):
            end=min(begin+32,len(images));x=images[begin:end].cuda();m=mask[begin:end].cuda()
            regp=predict(reg,'regression',x,m,eps[0,begin:end].cuda()).clamp(-1,1)
            target=fm16[geom][:,begin:end].cuda()
            den=3*m.sum((1,2,3))
            target_var=((target-target.mean(0)[None]).square()*m[None]).sum((2,3,4)).mean(0)/den
            e=eps[:,begin:end].cuda();e=e-e.mean(0)[None]
            evar=(e.square()*m[None]).sum((2,3,4)).mean(0)/den
            scale=(target_var/evar.clamp_min(1e-12)).sqrt()
            raw=regp[None]+m[None]*e*scale[None,:,None,None,None]
            values=scores(raw.clamp(-1,1),raw,x,m,metric)
            np.testing.assert_allclose(values['raw_variance'],target_var.cpu().numpy()/4,rtol=2e-5,atol=1e-8)
            for j in range(len(x)):
                rows.append({'index':begin+j,'geometry':geom,'model':'regression_white_noise','step':step,'nfe':1,
                             **{k:float(v[j]) for k,v in values.items()}})
    save_json(out/f'noise_control_step{step}_{split}_rows.json',rows)
    save_json(out/f'noise_control_step{step}_{split}_summary.json',aggregate(rows))
    return rows


@torch.no_grad()
def reference(root,out,net,step,images,cfg):
    count=cfg['reference_images'];masks_all=evaluation_masks(len(images),'eval')
    eps=noise_bank(images.shape,cfg['noise_seed'])[0,:count].cuda();x=images[:count].cuda()
    rows=[]
    for geom,mask in masks_all.items():
        m=mask[:count].cuda();den=3*m.sum((1,2,3))
        ref=predict(net,'fm',x,m,eps,128)
        for nfe in [1,2,4,8,16,32,64]:
            pred=predict(net,'fm',x,m,eps,nfe)
            rmse=((((pred-ref)/2).square()*m).sum((1,2,3))/den).sqrt()
            for i in range(count):rows.append({'index':i,'step':step,'geometry':geom,'nfe':nfe,
                                               'rmse_to_128':float(rmse[i]),'within_001':bool(rmse[i]<=0.01)})
    save_json(out/f'reference_step{step}.json',rows)
    return rows


def check_metric_math(metric):
    # Non-degenerate independent fixture: two exact facts and NumPy energy U-statistic.
    gen=torch.Generator().manual_seed(5)
    x=torch.rand((2,3,64,64),generator=gen).cuda()*2-1
    m=masks(2,'large',6).cuda()
    e=torch.randn((4,2,3,64,64),generator=gen).cuda()*.1
    p=(x[None]+m[None]*e).clamp(-1,1)
    r=scores(p,p,x,m,metric)
    aa=p.cpu().numpy();xx=x.cpu().numpy();mm=m.cpu().numpy();es=[]
    for b in range(2):
        vectors=np.array([(aa[i,b]-xx[b])[np.broadcast_to(mm[b]>0,aa[i,b].shape)]/2 for i in range(4)])
        truthdist=np.linalg.norm(vectors,axis=1).mean()/np.sqrt(vectors.shape[1])
        pairdist=sum(np.linalg.norm(vectors[i]-vectors[j]) for i,j in itertools.combinations(range(4),2))/(12*np.sqrt(vectors.shape[1]))
        es.append(truthdist-pairdist)
    np.testing.assert_allclose(es,r['energy_score'],atol=2e-7)
    identity=metric(x,x).flatten()
    assert torch.all(identity==0)
    repeated=x[None].expand(4,-1,-1,-1,-1)
    z=scores(repeated,repeated,x,m,metric)
    assert np.all(z['energy_score']==0) and np.all(z['ensemble_variance']==0)
    return {'passed':True,'checks':['LPIPS_identity','deterministic_ensemble_zero_variance',
                                  'MSE_mean_variance_identity','independent_numpy_energy_score_fixture']}


def extend(root,out,cfg,kind,cache,metric,startall):
    ckpt=root/'runs/pilot_seed0_v1'/f'{kind}_step2000.pt'
    c=torch.load(ckpt,weights_only=False,map_location='cpu')
    net=UNet().cuda();net.load_state_dict(c['model'])
    ema=copy.deepcopy(net).eval();ema.load_state_dict(c['ema'])
    opt=torch.optim.AdamW(net.parameters(),lr=cfg['learning_rate'],weight_decay=.0001)
    opt.load_state_dict(c['optimizer'])
    for group in opt.param_groups:
        assert abs(group['lr']-cfg['learning_rate'])<1e-12
    train=cache['train'].cuda().float()/127.5-1
    bank=torch.cat([masks(1024,g,300+i) for i,g in enumerate(['scattered','large'])]).cuda()
    dgen=torch.Generator(device='cuda').manual_seed(cfg['new_training_data_stream_seed'])
    ngen=torch.Generator(device='cuda').manual_seed(cfg['new_training_noise_stream_seed'])
    torch.cuda.reset_peak_memory_stats();window=[];clock=time.perf_counter();stream=hashlib.sha256()
    for step in range(2001,10001):
        if time.perf_counter()-startall>cfg['wall_seconds']:raise TimeoutError('v2 wall budget')
        ix=torch.randint(len(train),(64,),device='cuda',generator=dgen)
        flip=torch.rand((64,1,1,1),device='cuda',generator=dgen)<.5
        mi=torch.randint(1024,(64,),device='cuda',generator=dgen)
        geom=(torch.arange(64,device='cuda')%2)*1024
        # Stream identity binds each draw, independently of model/noise RNG.
        for v in [ix,flip,mi]:stream.update(v.cpu().numpy().tobytes())
        x=train[ix];x=torch.where(flip,x.flip(-1),x);m=bank[mi+geom];y=x*(1-m)
        if kind=='fm':
            e=torch.randn(x.shape,device='cuda',generator=ngen)
            t=torch.rand(64,device='cuda',generator=ngen)
            z=y+m*((1-t[:,None,None,None])*e+t[:,None,None,None]*x);target=x-e
        else:z=y;t=torch.zeros(64,device='cuda');target=x
        opt.zero_grad(set_to_none=True)
        with autocast():
            p=net(z,y,m,t)
            loss=((p.float()-target).square()*m).sum()/(3*m.sum())
        if not torch.isfinite(loss):raise ValueError('nonfinite loss')
        loss.backward();norm=nn.utils.clip_grad_norm_(net.parameters(),1.0,error_if_nonfinite=True);opt.step()
        with torch.no_grad():
            for ep,np_ in zip(ema.parameters(),net.parameters()):ep.lerp_(np_,.005)
        window.append(float(loss))
        if step%200==0:
            row={'model':kind,'step':step,'loss_mean_last200':float(np.mean(window)),
                 'gradient_norm':float(norm),'seconds':time.perf_counter()-clock,
                 'batch_stream_sha256':stream.hexdigest(),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
            with (out/f'{kind}_train.jsonl').open('a') as f:f.write(json.dumps(row)+'\n')
            save_json(out/'state.json',{'status':'training',**row});print('TRAIN_V2',json.dumps(row),flush=True);window=[]
        if step in cfg['dev_snapshots']:
            torch.save({'model':net.state_dict(),'ema':ema.state_dict(),'optimizer':opt.state_dict(),
                        'step':step,'kind':kind,'config':cfg,'data_generator_state':dgen.get_state(),
                        'noise_generator_state':ngen.get_state(),'torch_rng':torch.get_rng_state(),
                        'cuda_rng':torch.cuda.get_rng_state_all(),'batch_stream_sha256':stream.hexdigest()},out/f'{kind}_step{step}.pt')
            evaluate(root,out,ema,kind,step,cache['dev'].float()/127.5-1,'dev',metric,cfg,
                     nfes=[1] if kind=='regression' else [1,4,16])
    return ema


def run(root):
    cfg=json.loads((root/'followup_protocol.json').read_text())
    out=root/'runs'/cfg['run']
    if out.exists():raise RuntimeError('Refusing to overwrite existing v2 run')
    old=root/'runs/pilot_seed0_v1';identity=json.loads((old/'identity.json').read_text())
    assert identity['source_sha256']==digest(root/'pilot.py')
    assert identity['data_manifest_sha256']==digest(root/'data_manifest.json')
    manifest=json.loads((root/'data_manifest.json').read_text())
    assert manifest['source_manifest_sha256']==digest('/data/LGFNet/manifests/splits.csv')
    assert manifest['cache_sha256']==digest(root/'prepared.pt')
    out.mkdir(parents=True);save_json(out/'protocol.json',cfg)
    seed_all(0);torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    cache=torch.load(root/'prepared.pt',weights_only=True)
    metric=lpips.LPIPS(net='squeeze',version='0.1').cuda().eval()
    for p in metric.parameters():p.requires_grad_(False)
    save_json(out/'metric_checks.json',check_metric_math(metric))
    save_json(out/'identity.json',{'followup_source_sha256':digest(__file__),
       'original_pilot_source_sha256':digest(root/'pilot.py'),'protocol_sha256':digest(root/'followup_protocol.json'),
       'data_manifest_sha256':digest(root/'data_manifest.json'),
       'initial_checkpoint_sha256':{k:digest(old/f'{k}_step2000.pt') for k in ['regression','fm']},
       'lpips_trunk_sha256':digest('/data/geostep_cache/torch/hub/checkpoints/squeezenet1_1-b8a52dc0.pth'),
       'lpips_calibration_sha256':digest('/data/geostep_packages/lpips/weights/v0.1/squeeze.pth')})
    start=time.perf_counter();images=cache['eval'].float()/127.5-1
    initial_reg=load_net(old/'regression_step2000.pt')
    evaluate(root,out,initial_reg,'regression',2000,images,'eval',metric,cfg)
    initial_fm=load_net(old/'fm_step2000.pt')
    _,_,_,fm16=evaluate(root,out,initial_fm,'fm',2000,images,'eval',metric,cfg)
    white_noise_control(out,initial_reg,2000,images,'eval',metric,cfg,fm16)
    reference(root,out,initial_fm,2000,images,cfg)
    del initial_reg,initial_fm,fm16
    final_reg=extend(root,out,cfg,'regression',cache,metric,start)
    evaluate(root,out,final_reg,'regression',10000,images,'eval',metric,cfg)
    final_fm=extend(root,out,cfg,'fm',cache,metric,start)
    _,_,_,fm16=evaluate(root,out,final_fm,'fm',10000,images,'eval',metric,cfg)
    white_noise_control(out,final_reg,10000,images,'eval',metric,cfg,fm16)
    reference(root,out,final_fm,10000,images,cfg)
    histories={k:[json.loads(r) for r in (out/f'{k}_train.jsonl').read_text().splitlines()] for k in ['regression','fm']}
    assert [r['batch_stream_sha256'] for r in histories['regression']]==[r['batch_stream_sha256'] for r in histories['fm']]
    summary={'status':'complete','new_optimizer_updates':16000,'cumulative_main_updates':20000,
             'wall_seconds':time.perf_counter()-start,'metric_checks_passed':True,
             'matched_training_data_stream':True,'seeds':[0],
             'metric_groups':sum([json.loads(p.read_text()) for p in sorted(out.glob('*_eval_summary.json'))],[]),
             'limits':['exploratory reused validation','one seed','four noise samples','64x64 Pet only',
                       'LPIPS is paired distance, not distribution realism','energy score not semantic proof',
                       'warm-start continuation with new explicit RNG streams; original RNG not saved']}
    save_json(out/'summary.json',summary);save_json(out/'state.json',{'status':'complete','new_optimizer_updates':16000})
    print('FOLLOWUP_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);a=p.parse_args()
    try:run(a.root)
    except Exception as exc:
        save_json(a.root/'runs/followup_seed0_v2/failure.json',{'type':type(exc).__name__,'message':str(exc)})
        raise
