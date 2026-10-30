"""Verify each bounded candidate and selected seed without changing prior runs."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from pilot import autocast,digest,save_json
from capacity15_models import LargeUNet
from perceptual15_loss import PerceptualLoss,terms
from train_matched import make_probes,lr_at

def read(p):return json.loads(Path(p).read_text())
def run(root,tag,seed):
    parent=root/'runs/perceptual_regression_30k_v10';sd=parent/tag/f'seed{seed}';cfg=read(sd/'training_config.json');identity=read(sd/'identity.json');frozen=read(parent/'frozen_plan.json');cal=read(parent/'calibration.json')
    assert read(sd/'summary.json')['status']=='complete'
    for name,sha in frozen['sources'].items():assert digest(root/name)==sha
    assert digest(parent/'calibration.json')==frozen['calibration_sha256']==identity['calibration_sha256']
    assert digest(cal['vgg_weight_path'])==cal['vgg_weight_sha256']==identity['vgg_weight_sha256']
    for rel,sha in frozen['old_frozen_artifacts'].items():assert digest(root/rel)==sha
    assert cfg['alpha_tag']==tag and cfg['perceptual_coefficient']==cfg['alpha']*cal['feature_scale']
    assert read(sd/'metric_checks.json')['passed']
    md=root/'matched_masks_v4';mi=read(md/'identity.json');assert digest(md/'mask_bank.pt')==mi['mask_bank_sha256']
    assert digest(root/'prepared.pt')==read(root/'data_manifest.json')['cache_sha256']
    assert identity['mask_identity_sha256']==digest(md/'identity.json')
    supported=read(md/'supported_indices.json');cache=torch.load(root/'prepared.pt',weights_only=True);cache={s:cache[s][supported[s]]for s in ['train','dev']};bank=torch.load(md/'mask_bank.pt',weights_only=True)
    fm=root/'runs'/cfg['fm_runs'][str(seed)]/f'seed{seed}'
    initial=torch.load(sd/'initial_weights.pt',weights_only=True,map_location='cpu');original=torch.load(fm/'initial_weights.pt',weights_only=True,map_location='cpu')
    assert initial.keys()==original.keys()and all(torch.equal(v,original[k])for k,v in initial.items())
    assert sum(v.numel()for k,v in initial.items()if k!='freq')==14986435
    history=[json.loads(s)for s in (sd/'train.jsonl').read_text().splitlines()];baseline=[json.loads(s)for s in (fm/'train.jsonl').read_text().splitlines()]
    assert [r['step']for r in history]==list(range(200,30001,200))
    assert [r['batch_stream_sha256']for r in history]==[r['batch_stream_sha256']for r in baseline]
    assert np.isfinite([[r['loss_mean_last200'],r['gradient_norm']]for r in history]).all()
    for r in history:
        assert abs(r['learning_rate']-lr_at(r['step']))<1e-12
        d=r['component_means_last200'];assert abs(d['mse']+d['weighted_feature']-d['total'])<2e-6
        assert abs(d['feature_l1']*cfg['perceptual_coefficient']-d['weighted_feature'])<2e-6
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    checkpoints=[];final=None;n=len(cache['dev']);assert n==124
    for step in [10000,20000,30000]:
        p=sd/f'regression_step{step}.pt';c=torch.load(p,weights_only=False,map_location='cpu');assert c['step']==step and c['seed']==seed and c['config']==cfg
        assert set(int(v['step'])for v in c['optimizer']['state'].values())=={step}
        assert all(torch.isfinite(v).all()for key in ['model','ema']for v in c[key].values())
        assert {k:tuple(v.shape)for k,v in c['model'].items()}=={k:tuple(v.shape)for k,v in initial.items()}
        reference=torch.load(fm/f'fm_step{step}.pt',weights_only=False,map_location='cpu');assert torch.equal(c['data_generator_state'],reference['data_generator_state']);del reference
        assert len(read(sd/f'regression_step{step}_curve_rows.json'))==n*2
        assert len(read(sd/f'regression_step{step}_loss_rows.json'))==(128+n)*4
        objective=read(sd/f'regression_step{step}_objective_rows.json');assert len(objective)==(128+n)*4
        assert all(abs(r['mse']+r['weighted_feature']-r['total'])<2e-6 for r in objective)
        checkpoints.append({'seed':seed,'alpha_tag':tag,'step':step,'sha256':digest(p),'bytes':p.stat().st_size})
        if step==30000:final=c
        else:del c
    gen=torch.Generator(device='cuda').manual_seed(100+1000*seed);stream=hashlib.sha256();hist={r['step']:r for r in history}
    for step in range(1,30001):
        if step==2001:gen.manual_seed(202610213+1000*seed)
        for draw in [torch.randint(len(cache['train']),(64,),device='cuda',generator=gen),torch.rand((64,1,1,1),device='cuda',generator=gen)<.5,torch.randint(16,(64,),device='cuda',generator=gen)]:stream.update(draw.cpu().numpy().tobytes())
        if step%200==0:assert stream.hexdigest()==hist[step]['batch_stream_sha256']
    assert torch.equal(gen.get_state(),final['data_generator_state'])
    # Independently recompute MSE in NumPy and reconstruct the three VGG feature losses.
    net=LargeUNet().cuda().eval();net.load_state_dict(final['ema']);perceptual=PerceptualLoss().cuda().eval();probes,_=make_probes(cache,bank)
    recorded=read(sd/'regression_step30000_objective_rows.json');checked=0;max_error=0.
    for split,data in probes.items():
        x=data['images'][:32].cuda()
        for geom,mask in data['masks'].items():
            m=mask[:32].cuda();y=x*(1-m)
            with torch.no_grad(),autocast():
                pred=net(y,y,m,torch.zeros(len(x),device='cuda')).float();pf=perceptual.encode(y+m*pred);tf=perceptual.encode(x)
            pp=pred.cpu().numpy();xx=x.cpu().numpy();mm=m.cpu().numpy();mse=(((pp-xx)**2)*mm).sum((1,2,3))/(3*mm.sum((1,2,3)))
            feature=np.stack([np.abs(p.float().cpu().numpy()-t.float().cpu().numpy()).mean((1,2,3))for p,t in zip(pf,tf)]).mean(0)
            for i in range(8):
                row=next(r for r in recorded if r['variant']=='ema'and r['split']==split and r['geometry']==geom and r['index']==i)
                for key,values in [('mse',mse),('feature_l1',feature),('total',mse+cfg['perceptual_coefficient']*feature)]:
                    max_error=max(max_error,abs(float(values[i])-row[key]));np.testing.assert_allclose(values[i],row[key],atol=5e-4,rtol=.005)
                checked+=1
    rows=read(sd/'regression_step30000_final_rows.json');assert len(rows)==n*8
    assert len({(r['geometry'],r['image_index'],r['mask_pair'])for r in rows})==len(rows)
    assert np.isfinite([[r['lpips_truth'],r['sample_psnr'],r['sample_mse']]for r in rows]).all()
    assert all(r['ensemble_variance']==0 for r in rows)
    examples=np.load(sd/'regression_step30000_final_examples.npz');truth=cache['dev'][:3].float().numpy()/127.5-1
    for gi,g in enumerate(['large','fragmented']):
        arr=examples[g+'_1'];m=bank['dev'][:3,0,gi,None].numpy();assert arr.shape==(4,3,3,64,64)
        assert np.max(abs((arr-truth[None])*(1-m[None])))<2e-7
    save_json(sd/'checkpoint_manifest.json',checkpoints)
    result={'passed':True,'alpha_tag':tag,'seed':seed,'optimizer_updates':30000,'network_parameters':14986435,'initial_weights_equal_fm':True,'data_mask_flip_stream_equal_fm':True,'data_rng_replayed_updates':30000,'frozen_vgg_weights':cal['vgg_weight_sha256'],'train_images':2812,'dev_images':124,'final_rows':len(rows),'objective_probe_values_independently_checked':checked,'max_absolute_objective_recheck_error':max_error,'known_pixels_preserved':True,'old_artifacts_unchanged':True,'post_test_supplement':True,'test_images_accessed_by_training_and_selection':0,'numerical_recheck_tolerance':'NumPy MSE and feature L1, batch32 BF16, atol5e-4/rtol0.005','verifier_sha256':digest(__file__)}
    save_json(sd/'verification.json',result);print('VERIFIED',json.dumps(result),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--seed',type=int,required=True);p.add_argument('--alpha-tag',required=True);a=p.parse_args();run(a.root,a.alpha_tag,a.seed)
