"""Fixed-budget FM seeds on pre-audited, image-specific matched mask pairs."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import lpips
import numpy as np
import torch
from torch import nn

from pilot import UNet,autocast,digest,predict,save_json,seed_all
from followup import noise_bank,scores,check_metric_math
from continue30k import loss_probe,summarize_loss,array_hash


def lr_at(step):
    if step<=2000:
        return 2e-4*min(1.,step/100)*(.2+.8*.5*(1+math.cos(math.pi*step/2000)))
    if step<=10000:return 4e-5
    if step<=11000:return 4e-5+(step-10000)/1000*4e-5
    return 1e-5+.5*7e-5*(1+math.cos(math.pi*(step-11000)/19000))


def make_probes(cache,masks):
    gen=torch.Generator().manual_seed(202610216)
    ix=torch.randperm(len(cache['train']),generator=gen)[:128]
    probes={}
    for split,images,mm in [('train_probe',cache['train'][ix],masks['train'][ix,0]),
                            ('dev',cache['dev'],masks['dev'][:,0])]:
        gen=torch.Generator().manual_seed(202610215)
        images=images.float()/127.5-1
        bins=[]
        for low,high in zip([0.,.2,.4,.6,.8],[.2,.4,.6,.8,1.]):
            t=low+(high-low)*torch.rand(len(images),generator=gen)
            eps=torch.randn(images.shape,generator=gen)
            bins.append((low,high,t,eps))
        probes[split]={'images':images,'masks':{'large':mm[:,0,None].float(),
                         'fragmented':mm[:,1,None].float()},'bins':bins}
    return probes,ix


@torch.no_grad()
def evaluate(net,cache,maskbank,out,cfg,seed,step,all_pairs):
    net.eval()
    original=cache['dev'].float()/127.5-1
    pairs=cfg['dev_pairs_per_image'] if all_pairs else 1
    x=original[:,None].expand(-1,pairs,-1,-1,-1).reshape(-1,3,64,64)
    # A master [noise,image,mask-pair,C,H,W] bank makes curve pair0 exactly
    # the same condition/noise as pair0 in the final four-pair assessment.
    full=noise_bank((len(original),cfg['dev_pairs_per_image'],3,64,64),cfg['noise_seed'])
    eps=full[:,:,:pairs].reshape(4,-1,3,64,64)
    rows=[];summaries=[];examples={}
    for gi,geometry in enumerate(['large','fragmented']):
        mm=maskbank['dev'][:,:pairs,gi,None].reshape(-1,1,64,64).float()
        for nfe in cfg['nfe']:
            group=[]
            for begin in range(0,len(x),32):
                end=min(begin+32,len(x));b=end-begin
                xb=x[begin:end].cuda();m=mm[begin:end].cuda();e=eps[:,begin:end].cuda()
                xr=xb[None].expand(4,-1,-1,-1,-1).reshape(4*b,3,64,64)
                mr=m[None].expand(4,-1,-1,-1,-1).reshape(4*b,1,64,64)
                raw=predict(net,'fm',xr,mr,e.reshape(4*b,3,64,64),nfe).reshape(4,b,3,64,64)
                pred=raw.clamp(-1,1)
                values=scores(pred,raw,xb,m,evaluate.metric)
                for j in range(b):
                    condition=begin+j
                    row={'model':'fm','seed':seed,'step':step,'geometry':geometry,'nfe':nfe,
                         'condition_index':condition,'image_index':condition//pairs,'mask_pair':condition%pairs,
                         **{k:float(v[j]) for k,v in values.items()}}
                    rows.append(row);group.append(row)
                if begin==0:
                    examples[f'{geometry}_{nfe}']=pred[:,[0,pairs,2*pairs]].cpu().numpy()
            summary={'model':'fm','seed':seed,'step':step,'geometry':geometry,'nfe':nfe,
                     'images':len(original),'mask_pairs':pairs,'conditions':len(x),
                     **{k:float(np.mean([r[k] for r in group])) for k in values}}
            summaries.append(summary)
            print('TASK',json.dumps(summary),flush=True)
    stage='final' if all_pairs else 'curve'
    stem=f'fm_step{step}_{stage}'
    save_json(out/f'{stem}_rows.json',rows)
    save_json(out/f'{stem}_summary.json',summaries)
    if all_pairs:np.savez_compressed(out/f'{stem}_examples.npz',**examples)
    return summaries


def assess(net,ema,cache,maskbank,out,cfg,seed,step,probes):
    rows=[]
    for variant,model in [('online',net),('ema',ema)]:
        rows.extend(loss_probe(model,'fm',variant,step,probes))
    for row in rows:row['seed']=seed
    save_json(out/f'fm_step{step}_loss_rows.json',rows)
    save_json(out/f'fm_step{step}_loss_summary.json',summarize_loss(rows))
    evaluate(ema,cache,maskbank,out,cfg,seed,step,False)
    net.train()


def run(root,seed):
    cfg=json.loads((root/'matched_mask_protocol.json').read_text())
    assert seed in cfg['seeds']
    parent=root/'runs'/cfg['run'];out=parent/f'seed{seed}'
    if out.exists():raise RuntimeError('Refusing to overwrite existing matched seed')
    maskdir=root/'matched_masks_v4'
    audit=json.loads((maskdir/'mask_audit.json').read_text())
    assert audit['passed'] and not audit['probe_only']
    identity=json.loads((maskdir/'identity.json').read_text())
    assert identity['protocol_sha256']==digest(root/'matched_mask_protocol.json')
    assert identity['source_sha256']==digest(root/'matched_masks.py')
    assert identity['mask_bank_sha256']==digest(maskdir/'mask_bank.pt')
    prior=json.loads((root/'runs/continuation_seed0_30k_v3/identity.json').read_text())
    assert digest(root/'pilot.py')==prior['pilot_sha256']
    manifest=json.loads((root/'data_manifest.json').read_text())
    assert digest(root/'prepared.pt')==manifest['cache_sha256']
    cache=torch.load(root/'prepared.pt',weights_only=True)
    maskbank=torch.load(maskdir/'mask_bank.pt',weights_only=True)
    assert identity['supported_indices_sha256']==digest(maskdir/'supported_indices.json')
    supported=json.loads((maskdir/'supported_indices.json').read_text())
    cache={s:cache[s][supported[s]] for s in ['train','dev']}
    assert tuple(maskbank['train'].shape)==(len(cache['train']),16,2,64,64)
    assert tuple(maskbank['dev'].shape)==(len(cache['dev']),4,2,64,64)
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    seed_all(seed)
    net=UNet().cuda();ema=copy.deepcopy(net).eval()
    assert sum(p.numel() for p in net.parameters())==cfg['network_parameters']
    opt=torch.optim.AdamW(net.parameters(),lr=2e-4,weight_decay=cfg['weight_decay'])
    out.mkdir(parents=True)
    if not (parent/'protocol.json').exists():save_json(parent/'protocol.json',cfg)
    else:assert json.loads((parent/'protocol.json').read_text())==cfg
    torch.save(net.state_dict(),out/'initial_weights.pt')
    save_json(out/'identity.json',{'seed':seed,'source_sha256':digest(__file__),
              'protocol_sha256':digest(root/'matched_mask_protocol.json'),
              'pilot_source_sha256':digest(root/'pilot.py'),
              'metric_source_sha256':digest(root/'followup.py'),
              'loss_probe_source_sha256':digest(root/'continue30k.py'),
              'mask_identity_sha256':digest(maskdir/'identity.json'),
              'supported_original_indices':supported,
              'data_manifest_sha256':digest(root/'data_manifest.json'),
              'initial_weights_sha256':digest(out/'initial_weights.pt'),
              'torch_version':torch.__version__,'gpu':torch.cuda.get_device_name()})
    metric=lpips.LPIPS(net='squeeze',version='0.1').cuda().eval()
    for p in metric.parameters():p.requires_grad_(False)
    evaluate.metric=metric
    save_json(out/'metric_checks.json',check_metric_math(metric))
    probes,probe_indices=make_probes(cache,maskbank)
    save_json(out/'probe_identity.json',{'train_indices':probe_indices.tolist(),
              'mask_hashes':{s:{g:array_hash(m) for g,m in data['masks'].items()} for s,data in probes.items()},
              'noise_seed':202610215})
    train=cache['train'].cuda().float()/127.5-1
    bank=maskbank['train'].cuda()
    dgen=torch.Generator(device='cuda').manual_seed(100+1000*seed)
    ngen=torch.Generator(device='cuda').manual_seed(200+1000*seed)
    stream=hashlib.sha256();window=[];start=time.perf_counter();train_seconds=0.
    torch.cuda.reset_peak_memory_stats()
    for step in range(1,30001):
        if time.perf_counter()-start>5400:raise TimeoutError('Matched seed wall budget exceeded')
        if step==2001:
            # Reproduce the original two-stage data/noise stream boundary.
            dgen.manual_seed(202610213+1000*seed);ngen.manual_seed(202610214+1000*seed)
        clock=time.perf_counter()
        ix=torch.randint(len(train),(64,),device='cuda',generator=dgen)
        flip=torch.rand((64,1,1,1),device='cuda',generator=dgen)<.5
        mi=torch.randint(16,(64,),device='cuda',generator=dgen)
        geo=1-torch.arange(64,device='cuda')%2
        for value in [ix,flip,mi]:stream.update(value.cpu().numpy().tobytes())
        x=train[ix];m=bank[ix,mi,geo][:,None].float()
        x=torch.where(flip,x.flip(-1),x);m=torch.where(flip,m.flip(-1),m)
        y=x*(1-m)
        e=torch.randn(x.shape,device='cuda',generator=ngen)
        t=torch.rand(64,device='cuda',generator=ngen)
        z=y+m*((1-t[:,None,None,None])*e+t[:,None,None,None]*x)
        target=x-e
        lr=lr_at(step)
        for group in opt.param_groups:group['lr']=lr
        opt.zero_grad(set_to_none=True)
        with autocast():loss=((net(z,y,m,t).float()-target).square()*m).sum()/(3*m.sum())
        if not torch.isfinite(loss):raise ValueError('Nonfinite matched training loss')
        loss.backward();norm=nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True);opt.step()
        with torch.no_grad():
            for ep,p in zip(ema.parameters(),net.parameters()):ep.lerp_(p,.005)
        window.append(float(loss));train_seconds+=time.perf_counter()-clock
        if step%200==0:
            row={'seed':seed,'step':step,'learning_rate':lr,'loss_mean_last200':float(np.mean(window)),
                 'gradient_norm':float(norm),'training_seconds':train_seconds,
                 'batch_stream_sha256':stream.hexdigest(),'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
            with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            save_json(out/'state.json',{'status':'training',**row});print('TRAIN_MATCHED',json.dumps(row),flush=True);window=[]
        if step in cfg['checkpoints']:
            assert set(int(v['step']) for v in opt.state.values())=={step}
            torch.save({'model':net.state_dict(),'ema':ema.state_dict(),'optimizer':opt.state_dict(),
                        'seed':seed,'step':step,'config':cfg,'data_generator_state':dgen.get_state(),
                        'noise_generator_state':ngen.get_state(),'torch_rng':torch.get_rng_state(),
                        'cuda_rng':torch.cuda.get_rng_state_all(),'batch_stream_sha256':stream.hexdigest()},out/f'fm_step{step}.pt')
            assess(net,ema,cache,maskbank,out,cfg,seed,step,probes)
    evaluate(ema,cache,maskbank,out,cfg,seed,30000,True)
    summary={'status':'complete','seed':seed,'optimizer_updates':30000,'network_parameters':1747523,
             'training_seconds':train_seconds,'wall_seconds':time.perf_counter()-start,
             'original_network_unchanged':True,'same_mask_bank_across_seeds':True,
             'official_test_access':False,'final_evaluation_conditions_per_geometry':len(cache['dev'])*4,
             'noise_samples_per_condition':4,'limits':cfg['limitations']}
    save_json(out/'summary.json',summary);save_json(out/'state.json',summary)
    print('MATCHED_SEED_COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,required=True)
    a=p.parse_args()
    try:run(a.root,a.seed)
    except Exception as exc:
        dest=a.root/'runs/matched_geometry_30k_v4'/f'seed{a.seed}'
        save_json(dest/'failure.json',{'type':type(exc).__name__,'message':str(exc)})
        raise
