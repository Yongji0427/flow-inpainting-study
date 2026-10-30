"""Inference-only re-evaluation of the old Regression on the current dev/masks.
Training distributions and capacities differ: this is not a controlled method comparison.
"""
import json,time
from pathlib import Path
import numpy as np
import torch,lpips
from pilot import UNet,predict,digest,save_json
from followup import scores,check_metric_math
from compare_capacity import interval

@torch.no_grad()
def main():
    root=Path('/data/GeoStep');out=root/'runs/regression30k_reeval_matched_v6'
    if out.exists():raise RuntimeError('Refusing to overwrite existing re-evaluation')
    old=root/'runs/continuation_seed0_30k_v3';md=root/'matched_masks_v4'
    read=lambda p:json.loads(p.read_text())
    assert read(old/'verification.json')['passed']
    ident=read(old/'identity.json')
    assert digest(root/'pilot.py')==ident['pilot_sha256']
    assert digest(root/'followup.py')==ident['followup_sha256']
    checkpoint=old/'regression_step30000.pt';cp_hash=digest(checkpoint)
    assert cp_hash==read(old/'checkpoint_manifest.json')[checkpoint.name]['sha256']
    manifest=read(root/'data_manifest.json');assert digest(root/'prepared.pt')==manifest['cache_sha256']
    mi=read(md/'identity.json');assert digest(md/'mask_bank.pt')==mi['mask_bank_sha256']
    assert digest(md/'supported_indices.json')==mi['supported_indices_sha256']
    frozen=read(root/'runs/capacity15_30k_v6/seed0/identity.json')
    assert digest(md/'identity.json')==frozen['mask_identity_sha256']
    assert read(root/'runs/capacity15_30k_v6/verification.json')['passed']
    metric_hashes={'lpips_trunk_sha256':digest('/data/geostep_cache/torch/hub/checkpoints/squeezenet1_1-b8a52dc0.pth'),'lpips_calibration_sha256':digest('/data/geostep_packages/lpips/weights/v0.1/squeeze.pth')}
    assert metric_hashes==read(root/'runs/capacity15_30k_v6/verification.json')['lpips_weight_hashes_equal_prior_frozen_metric']
    supported=read(md/'supported_indices.json');cache=torch.load(root/'prepared.pt',weights_only=True)
    images=cache['dev'][supported['dev']].float()/127.5-1;bank=torch.load(md/'mask_bank.pt',weights_only=True)['dev'];assert len(images)==124
    c=torch.load(checkpoint,weights_only=False,map_location='cpu');assert c['step']==30000
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True
    net=UNet().cuda().eval();net.load_state_dict(c['ema']);params=sum(p.numel()for p in net.parameters());assert params==1747523
    initial={k:v.clone()for k,v in net.state_dict().items()}
    for p in net.parameters():p.requires_grad_(False)
    metric=lpips.LPIPS(net='squeeze',version='0.1').cuda().eval()
    for p in metric.parameters():p.requires_grad_(False)
    checks=check_metric_math(metric);assert checks['passed']
    out.mkdir();save_json(out/'metric_checks.json',checks)
    rows=[];max_mse_error=0.;start=time.perf_counter()
    x=images[:,None].expand(-1,4,-1,-1,-1).reshape(-1,3,64,64)
    for gi,geom in enumerate(['large','fragmented']):
        masks=bank[:,:,gi,None].reshape(-1,1,64,64).float()
        for begin in range(0,len(x),32):
            xb=x[begin:begin+32].cuda();m=masks[begin:begin+32].cuda()
            raw=predict(net,'regression',xb,m,torch.zeros_like(xb),1)
            pred=raw.clamp(-1,1)
            assert ((pred-xb)*(1-m)).abs().max()==0
            values=scores(pred[None].expand(4,-1,-1,-1,-1),raw[None].expand(4,-1,-1,-1,-1),xb,m,metric)
            pp=pred.cpu().numpy();xx=xb.cpu().numpy();mm=m.cpu().numpy()
            independent=((((pp-xx)/2)**2)*mm).sum((1,2,3))/(3*mm.sum((1,2,3)))
            max_mse_error=max(max_mse_error,float(np.max(np.abs(independent-values['sample_mse']))))
            np.testing.assert_allclose(independent,values['sample_mse'],rtol=2e-6,atol=2e-7)
            for j in range(len(xb)):
                condition=begin+j
                rows.append({'model':'regression','seed':0,'step':30000,'nfe':1,'geometry':geom,'image_index':condition//4,'mask_pair':condition%4,**{k:float(v[j])for k,v in values.items()}})
    assert len(rows)==992 and len({(r['geometry'],r['image_index'],r['mask_pair'])for r in rows})==992
    assert all(torch.equal(v,net.state_dict()[k])for k,v in initial.items())
    assert digest(checkpoint)==cp_hash
    save_json(out/'regression_rows.json',rows)
    baseline={(r['geometry'],r['image_index'],r['mask_pair']):r for r in rows}
    comparisons=[]
    for label,run in [('1.75M','matched_geometry_30k_v4'),('6.73M','wide_capacity_30k_v5'),('14.99M','capacity15_30k_v6')]:
        assert read(root/'runs'/run/'verification.json')['passed']
        fm_rows=read(root/'runs'/run/'seed0/fm_step30000_final_rows.json')
        candidate={(r['geometry'],r['image_index'],r['mask_pair']):r for r in fm_rows if r['nfe']==8}
        assert baseline.keys()==candidate.keys()
        for geom in ['large','fragmented','overall']:
            geoms=['large','fragmented']if geom=='overall'else[geom]
            for met,sign in [('lpips_truth',1),('sample_psnr',-1)]:
                a=np.array([np.mean([baseline[g,i,j][met]for g in geoms for j in range(4)])for i in range(124)])
                b=np.array([np.mean([candidate[g,i,j][met]for g in geoms for j in range(4)])for i in range(124)])
                comparisons.append({'fm_parameters':label,'fm_nfe':8,'regression_parameters':'1.75M','regression_nfe':1,'geometry':geom,'metric':met,'regression_mean':float(a.mean()),'fm_mean':float(b.mean()),'fm_improvement':interval(sign*(a-b))})
    save_json(out/'comparison.json',comparisons)
    save_json(out/'verification.json',{'passed':True,'optimizer_updates_this_task':0,'checkpoint_and_weights_unchanged':True,'checkpoint_sha256':cp_hash,'source_sha256':digest(__file__),'pilot_sha256':digest(root/'pilot.py'),'metric_source_sha256':digest(root/'followup.py'),'mask_identity_sha256':digest(md/'identity.json'),'data_manifest_sha256':digest(root/'data_manifest.json'),'lpips_weight_hashes':metric_hashes,'dev_images':124,'mask_pairs_per_image':4,'regression_conditions':len(rows),'known_pixels_preserved':True,'max_numpy_mse_error':max_mse_error,'official_test_images_accessed':0,'wall_seconds':time.perf_counter()-start,'limitations':['Regression trained on old v3 masks and 2924 training images; FM trained on v4 matched masks and 2812 training images','Regression 1.75M vs current FM14.99M; not equal capacity','Regression1 NFE vs FM8; not equal inference cost','Single seed; repeatedly used dev; image-bootstrap does not capture seed variation','No training or official-test evaluation in this task']})
    print(json.dumps(comparisons),flush=True)
if __name__=='__main__':main()
