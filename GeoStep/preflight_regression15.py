"""Zero-update numerical and provenance checks for the authorized control."""
import json,shutil,time
from pathlib import Path
import torch,numpy as np,lpips
from pilot import autocast,digest,predict,save_json,seed_all
from capacity15_models import LargeUNet
from train_matched import make_probes
from followup import scores,check_metric_math
from regression15_eval import evaluate

def main():
    root=Path('/data/GeoStep');read=lambda p:json.loads(p.read_text());cfg=read(root/'regression15_protocol.json');out=root/'runs'/cfg['run'];fm=root/'runs'/cfg['fm_run'];md=root/'matched_masks_v4'
    assert not (out/'seed0').exists()and not (out/'pipeline.pid').exists()
    assert read(fm/'verification.json')['passed']and read(fm/'pipeline_summary.json')['status']=='complete_verified'
    assert shutil.disk_usage(root).free>2*1024**3
    fmcfg=read(fm/'protocol.json')
    for key in ['network_parameters','channels','batch_size','steps_per_model','checkpoints','learning_rate_recipe','weight_decay','gradient_clip','ema_decay','missing_pixels','train_pairs_per_image','dev_pairs_per_image']:
        assert cfg[key]==fmcfg[key],key
    fm_identity=read(fm/'seed0/identity.json')
    for filename,key in [('capacity15_models.py','capacity_model_source_sha256'),('train_matched.py','baseline_training_source_sha256'),('pilot.py','pilot_source_sha256'),('followup.py','metric_source_sha256'),('continue30k.py','loss_probe_source_sha256')]:assert digest(root/filename)==fm_identity[key]
    assert digest(md/'identity.json')==fm_identity['mask_identity_sha256']
    maskid=read(md/'identity.json');assert digest(md/'mask_bank.pt')==maskid['mask_bank_sha256']
    assert digest(md/'supported_indices.json')==maskid['supported_indices_sha256']
    assert digest(root/'prepared.pt')==read(root/'data_manifest.json')['cache_sha256']
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    seed_all(0);net=LargeUNet();old=torch.load(fm/'seed0/initial_weights.pt',weights_only=True,map_location='cpu')
    assert sum(p.numel()for p in net.parameters())==14986435
    assert net.state_dict().keys()==old.keys()and all(torch.equal(v,old[k])for k,v in net.state_dict().items())
    del old
    indices=read(md/'supported_indices.json');cache=torch.load(root/'prepared.pt',weights_only=True);bank=torch.load(md/'mask_bank.pt',weights_only=True)
    cache={s:cache[s][indices[s]]for s in ['train','dev']};probes,ix=make_probes(cache,bank)
    original_probe=read(fm/'seed0/probe_identity.json');assert ix.tolist()==original_probe['train_indices']
    net=net.cuda();x=cache['train'][:64].float().cuda()/127.5-1;m=bank['train'][:64,0,0,None].float().cuda();y=x*(1-m);t=torch.zeros(64,device='cuda')
    torch.cuda.reset_peak_memory_stats()
    with autocast():pred=net(y,y,m,t).float();loss=((pred-x).square()*m).sum()/(3*m.sum())
    loss.backward();assert torch.isfinite(loss)and all(p.grad is None or torch.isfinite(p.grad).all()for p in net.parameters());peak=torch.cuda.max_memory_allocated()/2**30
    # Check the actual masked-image target and normalization, independently in NumPy.
    pp=pred.detach().cpu().numpy();xx=x.cpu().numpy();mm=m.cpu().numpy();independent=(((pp-xx)**2)*mm).sum()/(3*mm.sum());np.testing.assert_allclose(independent,float(loss),rtol=2e-6,atol=2e-7)
    net.zero_grad(set_to_none=True);net.eval();raw=predict(net,'regression',x[:8],m[:8],torch.zeros_like(x[:8]),1);assert torch.isfinite(raw).all()and ((raw-x[:8])*(1-m[:8])).abs().max()==0
    metric=lpips.LPIPS(net='squeeze',version='0.1').cuda().eval()
    for p in metric.parameters():p.requires_grad_(False)
    checks=check_metric_math(metric);assert checks['passed'];evaluate.metric=metric
    # Exercise the final evaluator on four images/four pairs, without writing seed0.
    out.mkdir(parents=True,exist_ok=True);smoke=out/'evaluation_smoke';smoke.mkdir(exist_ok=True)
    smoke_cfg={**cfg,'nfe':[1]};evaluate(net,{'dev':cache['dev'][:4]},{'dev':bank['dev'][:4]},smoke,smoke_cfg,0,0,True)
    rows=read(smoke/'regression_step0_final_rows.json');assert len(rows)==32
    assert all(r['nfe']==1 and r['ensemble_variance']==0 and r['lpips_pairwise']==0 for r in rows)
    # Freeze all FM inputs used by training, verification, comparison and timing.
    names=['identity.json','initial_weights.pt','train.jsonl','probe_identity.json','fm_step30000_final_rows.json']
    for step in cfg['checkpoints']:names += [f'fm_step{step}.pt',f'fm_step{step}_curve_rows.json']
    frozen={'seed0/'+name:digest(fm/'seed0'/name)for name in names}
    frozen.update({name:digest(fm/name)for name in ['verification.json','protocol.json','checkpoint_manifest.json','pipeline_summary.json']})
    save_json(out/'fm_frozen_artifacts.json',frozen)
    sources=['train_regression15.py','regression15_eval.py','verify_regression15.py','compare_regression15.py','capacity15_models.py','train_matched.py','pilot.py','followup.py','continue30k.py','compare_capacity.py','regression15_protocol.json','run_regression15_v7.sh','preflight_regression15.py']
    result={'passed':True,'optimizer_updates':0,'parameters':14986435,'initial_weights_identical_to_fm':True,'finite_forward_backward_batch64':True,'known_pixels_preserved':True,'independent_numpy_training_loss_checked':True,'deterministic_metric_and_evaluator_smoke_passed':True,'smoke_rows':32,'preflight_peak_allocated_gib':peak,'train_images':len(cache['train']),'dev_images':len(cache['dev']),'source_sha256':{name:digest(root/name)for name in sources},'fm_frozen_manifest_sha256':digest(out/'fm_frozen_artifacts.json'),'official_test_access':False}
    save_json(out/'preflight.json',result);print(json.dumps(result),flush=True)
if __name__=='__main__':main()
