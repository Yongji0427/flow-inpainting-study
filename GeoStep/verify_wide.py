"""Verify the single width-control run without altering the frozen v4 artifacts."""
import argparse,hashlib,json
from pathlib import Path
import numpy as np
import torch
from pilot import autocast,digest,save_json
from capacity_models import WideUNet as UNet
from train_matched import make_probes,lr_at

def read(p):return json.loads(Path(p).read_text())
def run(root):
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    out=root/'runs/wide_capacity_30k_v5';md=root/'matched_masks_v4'
    cfg=read(out/'protocol.json');mi=read(md/'identity.json');audit=read(md/'mask_audit.json')
    manifest=read(root/'data_manifest.json');assert digest(root/'prepared.pt')==manifest['cache_sha256']
    old_metric=read(root/'runs/followup_seed0_v2/identity.json')
    metric_hashes={'lpips_trunk_sha256':digest('/data/geostep_cache/torch/hub/checkpoints/squeezenet1_1-b8a52dc0.pth'),'lpips_calibration_sha256':digest('/data/geostep_packages/lpips/weights/v0.1/squeeze.pth')}
    assert all(value==old_metric[key]for key,value in metric_hashes.items())
    assert audit['passed'] and not audit['probe_only'] and read(md/'preflight.json')['passed']
    assert mi['source_sha256']==digest(root/'matched_masks.py')
    assert mi['protocol_sha256']==digest(root/'matched_mask_protocol.json')
    # Serialized copies may differ by a trailing newline; compare their contents.
    assert cfg==read(root/'wide_capacity_protocol.json')
    for filename,key in [('mask_bank.pt','mask_bank_sha256'),('pair_metadata.json','pair_metadata_sha256'),('supported_indices.json','supported_indices_sha256')]:assert digest(md/filename)==mi[key]
    supported=read(md/'supported_indices.json');cache=torch.load(root/'prepared.pt',weights_only=True)
    cache={s:cache[s][supported[s]]for s in ['train','dev']};bank=torch.load(md/'mask_bank.pt',weights_only=True)
    probes,ix=make_probes(cache,bank);n=len(cache['dev']);checkpoints=[];initials=[];checked=0;max_error=0.;seed_results=[]
    for seed in [0]:
        sd=out/f'seed{seed}';summ=read(sd/'summary.json');identity=read(sd/'identity.json')
        assert summ['status']=='complete' and summ['optimizer_updates']==30000
        assert summ['final_evaluation_conditions_per_geometry']==4*n
        for file,key in [('train_wide.py','source_sha256'),('capacity_models.py','capacity_model_source_sha256'),('train_matched.py','baseline_training_source_sha256'),('pilot.py','pilot_source_sha256'),('followup.py','metric_source_sha256'),('continue30k.py','loss_probe_source_sha256')]:assert digest(root/file)==identity[key]
        assert identity['mask_identity_sha256']==digest(md/'identity.json')
        assert identity['protocol_sha256']==digest(root/'wide_capacity_protocol.json')
        assert identity['supported_original_indices']==supported
        assert identity['data_manifest_sha256']==digest(root/'data_manifest.json')
        assert read(sd/'metric_checks.json')['passed']
        initials.append(torch.load(sd/'initial_weights.pt',weights_only=True,map_location='cpu'))
        history=[json.loads(s)for s in (sd/'train.jsonl').read_text().splitlines()]
        assert [r['step']for r in history]==list(range(200,30001,200))
        assert np.isfinite([[r['loss_mean_last200'],r['gradient_norm']]for r in history]).all()
        for row in history:assert abs(row['learning_rate']-lr_at(row['step']))<1e-12
        final=None
        for step in [10000,20000,30000]:
            path=sd/f'fm_step{step}.pt';c=torch.load(path,weights_only=False,map_location='cpu')
            assert c['step']==step and c['seed']==seed and c['config']==cfg
            assert set(int(v['step'])for v in c['optimizer']['state'].values())=={step}
            assert all(torch.isfinite(t).all()for key in ['model','ema']for t in c[key].values())
            assert {k:tuple(v.shape)for k,v in initials[-1].items()}=={k:tuple(v.shape)for k,v in c['model'].items()}
            checkpoints.append({'seed':seed,'step':step,'sha256':digest(path),'bytes':path.stat().st_size})
            rows=read(sd/f'fm_step{step}_loss_rows.json')
            assert len(rows)==(128+n)*2*2*5
            assert len({(r['variant'],r['split'],r['geometry'],r['index'],r['t_low'])for r in rows})==len(rows)
            assert np.isfinite([r['loss']for r in rows]).all()
            assert all(r['t_low']<=r['t']<r['t_high']for r in rows)
            tasks=read(sd/f'fm_step{step}_curve_rows.json');assert len(tasks)==n*2*6
            if step==30000:final=c
            else:del c
        final_rows=read(sd/'fm_step30000_final_rows.json');assert len(final_rows)==n*4*2*6
        assert len({(r['geometry'],r['nfe'],r['image_index'],r['mask_pair'])for r in final_rows})==len(final_rows)
        assert np.isfinite([[r[k]for k in ['sample_mse','sample_psnr','lpips_truth','energy_score']]for r in final_rows]).all()
        assert all(abs(r['sample_mse']-r['mean_prediction_mse']-r['ensemble_variance'])<2e-7 for r in final_rows)
        # Replay the actual CUDA data RNG independently of training and evaluation.
        gen=torch.Generator(device='cuda').manual_seed(100+1000*seed);stream=hashlib.sha256();hist={r['step']:r for r in history}
        for step in range(1,30001):
            if step==2001:gen.manual_seed(202610213+1000*seed)
            for draw in [torch.randint(len(cache['train']),(64,),device='cuda',generator=gen),torch.rand((64,1,1,1),device='cuda',generator=gen)<.5,torch.randint(16,(64,),device='cuda',generator=gen)]:stream.update(draw.cpu().numpy().tobytes())
            if step%200==0:assert stream.hexdigest()==hist[step]['batch_stream_sha256']
        assert torch.equal(gen.get_state(),final['data_generator_state'])
        net=UNet().cuda().eval();net.load_state_dict(final['ema']);recorded=read(sd/'fm_step30000_loss_rows.json')
        for split,data in probes.items():
            x=data['images'][:32].cuda()
            for geometry,mask in data['masks'].items():
                m=mask[:32].cuda();y=x*(1-m)
                for low,high,times,noise in data['bins']:
                    t=times[:32].cuda();e=noise[:32].cuda();z=y+m*((1-t[:,None,None,None])*e+t[:,None,None,None]*x)
                    with torch.no_grad(),autocast():pred=net(z,y,m,t).float().cpu().numpy()
                    target=(x-e).cpu().numpy();mm=m.cpu().numpy();values=((pred-target)**2*mm).sum((1,2,3))/(3*mm.sum((1,2,3)))
                    for i,value in enumerate(values[:8]):
                        row=next(r for r in recorded if r['variant']=='ema'and r['split']==split and r['geometry']==geometry and r['index']==i and r['t_low']==low)
                        max_error=max(max_error,abs(float(value)-row['loss']));np.testing.assert_allclose(value,row['loss'],atol=3e-4,rtol=.003);checked+=1
        examples=np.load(sd/'fm_step30000_final_examples.npz')
        truth=cache['dev'][:3].float().numpy()/127.5-1
        for gi,g in enumerate(['large','fragmented']):
            m=bank['dev'][:3,0,gi,None].numpy()
            for nfe in cfg['nfe']:
                arr=examples[f'{g}_{nfe}'];assert np.isfinite(arr).all();assert arr.shape==(4,3,3,64,64)
                assert np.max(abs((arr-truth[None])*(1-m[None])))<2e-7
        seed_results.append({'seed':seed,'optimizer_updates':30000,'data_rng_replayed_updates':30000,'final_rows':len(final_rows),'state':'verified'})
        del final,net
    baseline=root/'runs/matched_geometry_30k_v4/seed0'
    initial_small=torch.load(baseline/'initial_weights.pt',weights_only=True,map_location='cpu')
    for k,v in initial_small.items():
        if k.startswith('temb.') or k=='freq':assert torch.equal(v,initials[0][k])
    old_history=[json.loads(line)for line in (baseline/'train.jsonl').read_text().splitlines()]
    assert [r['batch_stream_sha256']for r in old_history]==[r['batch_stream_sha256']for r in history]
    assert sum(v.numel()for k,v in initials[0].items()if k!='freq')==6734723
    save_json(out/'checkpoint_manifest.json',checkpoints)
    result={'passed':True,'seeds':seed_results,'total_training_updates':30000,'same_mask_bank_as_baseline':True,'same_data_mask_noise_stream_as_baseline':True,'network_parameters':6734723,'train_images':len(cache['train']),'dev_images':n,'frozen_v4_mask_audit_reused':True,'numpy_loss_values_checked':checked,'max_numpy_loss_error':max_error,'known_pixels_preserved_in_saved_examples':True,'test_images_accessed_this_experiment':0,'lpips_weight_hashes_equal_prior_frozen_metric':metric_hashes,'verifier_sha256':digest(__file__),'numerics':'CPU normalization and batch32 BF16 forwards; NumPy independent losses atol3e-4/rtol0.003'}
    save_json(out/'verification.json',result);print(json.dumps(result),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);run(p.parse_args().root)
