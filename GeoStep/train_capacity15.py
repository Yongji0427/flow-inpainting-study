"""Width-only 30k training, using the immutable v4 data/probes/evaluation."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import time
import lpips
import numpy as np
import torch
from torch import nn
from pilot import autocast,digest,save_json,seed_all
from capacity15_models import LargeUNet
from train_matched import lr_at,make_probes,evaluate,assess,array_hash,check_metric_math


def run(root,seed):
    cfg=json.loads((root/'capacity15_protocol.json').read_text())
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
    baseline_identity=json.loads((root/'runs'/cfg['baseline_run']/f'seed{seed}'/'identity.json').read_text())
    for file,key in [('train_matched.py','source_sha256'),('pilot.py','pilot_source_sha256'),('followup.py','metric_source_sha256'),('continue30k.py','loss_probe_source_sha256')]:
        assert digest(root/file)==baseline_identity[key],f'Frozen baseline source changed: {file}'
    assert baseline_identity['mask_identity_sha256']==digest(maskdir/'identity.json')
    assert json.loads((root/'runs'/cfg['baseline_run']/'verification.json').read_text())['passed']
    seed_all(seed)
    net=LargeUNet().cuda();ema=copy.deepcopy(net).eval()
    assert sum(p.numel() for p in net.parameters())==cfg['network_parameters']
    opt=torch.optim.AdamW(net.parameters(),lr=2e-4,weight_decay=cfg['weight_decay'])
    out.mkdir(parents=True)
    if not (parent/'protocol.json').exists():save_json(parent/'protocol.json',cfg)
    else:assert json.loads((parent/'protocol.json').read_text())==cfg
    torch.save(net.state_dict(),out/'initial_weights.pt')
    save_json(out/'identity.json',{'seed':seed,'source_sha256':digest(__file__),
              'protocol_sha256':digest(root/'capacity15_protocol.json'),
              'capacity_model_source_sha256':digest(root/'capacity15_models.py'),
              'baseline_training_source_sha256':digest(root/'train_matched.py'),
              'pilot_source_sha256':digest(root/'pilot.py'),
              'metric_source_sha256':digest(root/'followup.py'),
              'loss_probe_source_sha256':digest(root/'continue30k.py'),
              'mask_identity_sha256':digest(maskdir/'identity.json'),
              'supported_original_indices':supported,
              'data_manifest_sha256':digest(root/'data_manifest.json'),
              'initial_weights_sha256':digest(out/'initial_weights.pt'),
              'torch_version':torch.__version__,'gpu':torch.cuda.get_device_name()})
    baseline_dir=root/'runs'/cfg['baseline_run']/f'seed{seed}'
    baseline_history={r['step']:r for r in [json.loads(line) for line in (baseline_dir/'train.jsonl').read_text().splitlines()]}
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
        if time.perf_counter()-start>cfg['wall_budget_seconds']:raise TimeoutError('Wide seed wall budget exceeded')
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
            assert row['batch_stream_sha256']==baseline_history[step]['batch_stream_sha256'],'Training data/mask stream differs from frozen baseline'
            with (out/'train.jsonl').open('a') as f:f.write(json.dumps(row,allow_nan=False)+'\n')
            save_json(out/'state.json',{'status':'training',**row});print('TRAIN_WIDE',json.dumps(row),flush=True);window=[]
        if step in cfg['checkpoints']:
            baseline_checkpoint=torch.load(baseline_dir/f'fm_step{step}.pt',weights_only=False,map_location='cpu')
            assert torch.equal(dgen.get_state(),baseline_checkpoint['data_generator_state'])
            assert torch.equal(ngen.get_state(),baseline_checkpoint['noise_generator_state'])
            del baseline_checkpoint
            assert set(int(v['step']) for v in opt.state.values())=={step}
            torch.save({'model':net.state_dict(),'ema':ema.state_dict(),'optimizer':opt.state_dict(),
                        'seed':seed,'step':step,'config':cfg,'data_generator_state':dgen.get_state(),
                        'noise_generator_state':ngen.get_state(),'torch_rng':torch.get_rng_state(),
                        'cuda_rng':torch.cuda.get_rng_state_all(),'batch_stream_sha256':stream.hexdigest()},out/f'fm_step{step}.pt')
            assess(net,ema,cache,maskbank,out,cfg,seed,step,probes)
    evaluate(ema,cache,maskbank,out,cfg,seed,30000,True)
    summary={'status':'complete','seed':seed,'optimizer_updates':30000,'network_parameters':14986435,
             'training_seconds':train_seconds,'wall_seconds':time.perf_counter()-start,
             'width_only_change':True,'inherited_original_forward':True,'same_mask_bank_across_seeds':True,
             'official_test_access':False,'final_evaluation_conditions_per_geometry':len(cache['dev'])*4,
             'noise_samples_per_condition':4,'limits':cfg['limitations']}
    save_json(out/'summary.json',summary);save_json(out/'state.json',summary)
    print('WIDE_SEED_COMPLETE',json.dumps(summary),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,default=0)
    a=p.parse_args()
    try:run(a.root,a.seed)
    except Exception as exc:
        dest=a.root/'runs/capacity15_30k_v6'/f'seed{a.seed}'
        # Never alter an already completed run after refusing an overwrite.
        if not (dest/'summary.json').exists():save_json(dest/'failure.json',{'type':type(exc).__name__,'message':str(exc)})
        raise
