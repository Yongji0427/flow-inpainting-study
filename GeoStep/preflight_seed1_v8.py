"""Seed1 pair preflight: no optimizer updates and no official-test access."""
import json
from pathlib import Path
import torch,lpips,hashlib
from pilot import seed_all,autocast,digest,save_json
from capacity15_models import LargeUNet
from train_matched import evaluate as fm_evaluate
from regression15_eval import evaluate as reg_evaluate
from official_test_v9 import verify_saved
from followup import check_metric_math
ROOT=Path('/data/GeoStep');OUT=ROOT/'runs/replication15_two_seeds_v8'
def read(p):return json.loads(Path(p).read_text())
def main():
    assert not OUT.exists(),'Refuse duplicate launch'
    assert not (ROOT/'runs/capacity15_seed1_30k_v8').exists()
    assert not (ROOT/'runs/regression15_seed1_30k_v8').exists()
    assert not (ROOT/'runs/official_test_two_seeds_v9').exists()
    for name in ['capacity15_30k_v6','regression15_30k_v7','matched_geometry_30k_v4']:
        assert read(ROOT/'runs'/name/'verification.json')['passed']
    fmold=read(ROOT/'runs/capacity15_30k_v6/seed0/identity.json')
    for name,key in [('capacity15_models.py','capacity_model_source_sha256'),('train_matched.py','baseline_training_source_sha256'),('pilot.py','pilot_source_sha256'),('followup.py','metric_source_sha256'),('continue30k.py','loss_probe_source_sha256')]:assert digest(ROOT/name)==fmold[key]
    assert digest(ROOT/'matched_masks_v4/identity.json')==fmold['mask_identity_sha256']
    mid=read(ROOT/'matched_masks_v4/identity.json');assert digest(ROOT/'matched_masks_v4/mask_bank.pt')==mid['mask_bank_sha256']
    assert digest(ROOT/'prepared.pt')==read(ROOT/'data_manifest.json')['cache_sha256']
    # Compare all configuration fields intended to remain fixed.
    for current,old in [('capacity15_seed1_protocol.json','capacity15_protocol.json'),('regression15_seed1_protocol.json','regression15_protocol.json')]:
        a=read(ROOT/current);b=read(ROOT/old)
        changes={k for k in a.keys()|b.keys()if a.get(k)!=b.get(k)}
        allowed={'run','scope','seeds','baseline_seed','limitations','fm_run','comparison_run'}
        assert changes<=allowed,changes
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    seed_all(1);net=LargeUNet();assert sum(p.numel()for p in net.parameters())==14986435
    baseline=torch.load(ROOT/'runs/matched_geometry_30k_v4/seed1/initial_weights.pt',weights_only=True,map_location='cpu')
    for k,v in baseline.items():
        if k.startswith('temb.')or k=='freq':assert torch.equal(v,net.state_dict()[k])
    old=torch.load(ROOT/'runs/capacity15_30k_v6/seed0/initial_weights.pt',weights_only=True,map_location='cpu')
    assert any(not torch.equal(v,old[k])for k,v in net.state_dict().items())
    seed_all(1);other=LargeUNet();assert all(torch.equal(v,other.state_dict()[k])for k,v in net.state_dict().items());del old,other,baseline
    ix=read(ROOT/'matched_masks_v4/supported_indices.json');cache=torch.load(ROOT/'prepared.pt',weights_only=True);cache={s:cache[s][ix[s]]for s in ['train','dev']};bank=torch.load(ROOT/'matched_masks_v4/mask_bank.pt',weights_only=True)
    # Reproduce the actual CUDA seed1 data/mask RNG without updating any model.
    gen=torch.Generator(device='cuda').manual_seed(1100);stream=hashlib.sha256();history={r['step']:r for r in [json.loads(s)for s in (ROOT/'runs/matched_geometry_30k_v4/seed1/train.jsonl').read_text().splitlines()]}
    for step in range(1,401):
        for draw in [torch.randint(len(cache['train']),(64,),device='cuda',generator=gen),torch.rand((64,1,1,1),device='cuda',generator=gen)<.5,torch.randint(16,(64,),device='cuda',generator=gen)]:stream.update(draw.cpu().numpy().tobytes())
        if step%200==0:assert stream.hexdigest()==history[step]['batch_stream_sha256']
    net=net.cuda();x=cache['train'][:64].float().cuda()/127.5-1;m=bank['train'][:64,0,0,None].float().cuda();y=x*(1-m)
    for kind in ['fm','regression']:
        net.zero_grad(set_to_none=True)
        if kind=='fm':
            e=torch.randn_like(x);t=torch.rand(64,device='cuda');z=y+m*((1-t[:,None,None,None])*e+t[:,None,None,None]*x);target=x-e
        else:t=torch.zeros(64,device='cuda');z=y;target=x
        with autocast():loss=((net(z,y,m,t).float()-target).square()*m).sum()/(3*m.sum())
        loss.backward();assert torch.isfinite(loss)and all(p.grad is None or torch.isfinite(p.grad).all()for p in net.parameters())
    net.zero_grad(set_to_none=True);net.eval();OUT.mkdir(parents=True)
    metric=lpips.LPIPS(net='squeeze',version='0.1').cuda().eval()
    for p in metric.parameters():p.requires_grad_(False)
    assert check_metric_math(metric)['passed'];fm_evaluate.metric=metric;reg_evaluate.metric=metric
    for kind,function,nfe in [('fm',fm_evaluate,8),('regression',reg_evaluate,1)]:
        dest=OUT/('smoke_'+kind);dest.mkdir()
        function(net,{'dev':cache['dev'][:4]},{'dev':bank['dev'][:4]},dest,{'dev_pairs_per_image':4,'noise_seed':202610225,'nfe':[nfe]},1,30000,True)
        verify_saved(dest,kind,cache['dev'][:4],bank['dev'][:4])
    names=['train_capacity15_seed1.py','verify_capacity15_seed1.py','train_regression15_seed1.py','verify_regression15_seed1.py','preflight_regression15_seed1.py','compare_regression15_seed1.py','capacity15_seed1_protocol.json','regression15_seed1_protocol.json','seed1_pipeline.py','official_test_v9.py','official_test_protocol_v9.json','preflight_seed1_v8.py','run_seed1_v8.sh','capacity15_models.py','pilot.py','train_matched.py','followup.py','continue30k.py','regression15_eval.py','compare_capacity.py','compare_regression15.py','matched_masks.py','matched_mask_protocol.json','protocol.json','data_manifest.json']
    plan={'optimizer_updates_before_launch':0,'official_test_images_opened_before_launch':0,'seeds':[0,1],'new_training_seed':1,'new_training_updates':60000,'sources':{name:digest(ROOT/name)for name in names},'test_protocol_sha256':digest(ROOT/'official_test_protocol_v9.json'),'old_run_identities':{name:digest(ROOT/'runs'/name/'verification.json')for name in ['capacity15_30k_v6','regression15_30k_v7']},'seed1_data_stream_first400_verified':True,'initial_seed1_pair_identical_and_different_from_seed0':True,'both_objectives_batch64_finite_backward':True,'dev_only_smoke_of_test_metric_verifier_passed':True}
    save_json(OUT/'frozen_plan.json',plan);save_json(OUT/'preflight.json',{'passed':True,**plan})
    source=OUT/'source';source.mkdir()
    for name,sha in plan['sources'].items():(source/name).write_bytes((ROOT/name).read_bytes())
    print('PREFLIGHT_PASSED',json.dumps({'optimizer_updates':0,'official_test_images_opened':0,'new_training_updates':60000,'sources_frozen':len(names),'frozen_plan_sha256':digest(OUT/'frozen_plan.json')}),flush=True)
if __name__=='__main__':main()
