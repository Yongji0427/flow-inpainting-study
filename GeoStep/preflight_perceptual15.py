"""Zero-update loss/gradient/provenance checks before bounded post-test training."""
import hashlib,json,shutil,time
from pathlib import Path
import numpy as np
import torch,lpips
from torchvision.models import VGG16_Weights
from pilot import seed_all,autocast,predict,digest,save_json
from capacity15_models import LargeUNet
from perceptual15_loss import PerceptualLoss,terms
from regression15_eval import evaluate
from followup import check_metric_math
from official_test_v9 import verify_saved
ROOT=Path('/data/GeoStep');OUT=ROOT/'runs/perceptual_regression_30k_v10'
def read(p):return json.loads(Path(p).read_text())
def main():
    assert not OUT.exists(),'Refuse overwriting prior perceptual experiment'
    assert shutil.disk_usage(ROOT).free>5*1024**3
    cfg=read(ROOT/'perceptual15_protocol_v10.json');old=read(ROOT/'regression15_protocol.json')
    for k in ['batch_size','network_parameters','channels','steps_per_model','checkpoints','learning_rate_recipe','weight_decay','gradient_clip','ema_decay','missing_pixels','train_pairs_per_image','dev_pairs_per_image']:
        assert cfg[k]==old[k],k
    md=ROOT/'matched_masks_v4';mid=read(md/'identity.json');assert digest(md/'mask_bank.pt')==mid['mask_bank_sha256']
    assert digest(md/'supported_indices.json')==mid['supported_indices_sha256'];assert digest(ROOT/'prepared.pt')==read(ROOT/'data_manifest.json')['cache_sha256']
    for run in ['capacity15_30k_v6','capacity15_seed1_30k_v8','regression15_30k_v7','regression15_seed1_30k_v8']:
        assert read(ROOT/'runs'/run/'verification.json')['passed']
    for seed in [0,1]:
        fm=ROOT/'runs'/cfg['fm_runs'][str(seed)]/f'seed{seed}';identity=read(fm/'identity.json')
        assert identity['mask_identity_sha256']==digest(md/'identity.json')
        for name,key in [('capacity15_models.py','capacity_model_source_sha256'),('pilot.py','pilot_source_sha256'),('train_matched.py','baseline_training_source_sha256'),('followup.py','metric_source_sha256'),('continue30k.py','loss_probe_source_sha256')]:assert digest(ROOT/name)==identity[key]
        seed_all(seed);net=LargeUNet();reference=torch.load(fm/'initial_weights.pt',weights_only=True,map_location='cpu');assert all(torch.equal(v,reference[k])for k,v in net.state_dict().items())
    del reference,net
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;seed_all(0);net=LargeUNet().cuda()
    perceptual=PerceptualLoss().cuda().eval();assert all(not p.requires_grad for p in perceptual.parameters())
    weight=Path('/data/geostep_cache/torch/hub/checkpoints/vgg16-397923af.pth');weight_hash=digest(weight);assert weight_hash.startswith('397923af')
    ix=read(md/'supported_indices.json');cache=torch.load(ROOT/'prepared.pt',weights_only=True);cache={s:cache[s][ix[s]]for s in ['train','dev']};bank=torch.load(md/'mask_bank.pt',weights_only=True)
    x=cache['train'][:64].float().cuda()/127.5-1;geo=1-torch.arange(64)%2;m=bank['train'][torch.arange(64),0,geo,None].float().cuda()
    with torch.no_grad():
        initial=terms(net,perceptual,x,m,0.);pixel=float(initial['mse'].mean());feature=float(initial['feature_l1'].mean())
    assert pixel>0 and feature>0 and np.isfinite([pixel,feature]).all();scale=pixel/feature
    gradients=[];durations=[];torch.cuda.reset_peak_memory_stats()
    for tag,alpha in cfg['candidates'].items():
        net.zero_grad(set_to_none=True);torch.cuda.synchronize();start=time.perf_counter();values=terms(net,perceptual,x,m,alpha*scale);loss=values['total'].mean();loss.backward();torch.cuda.synchronize();durations.append(time.perf_counter()-start)
        assert torch.isfinite(loss)and all(p.grad is None or torch.isfinite(p.grad).all()for p in net.parameters())
        assert all(p.grad is None for p in perceptual.parameters());gradients.append(float(net.out[-1].weight.grad.norm()))
        assert gradients[-1]>0
    # Identical image gives exactly zero feature distance. Composition blocks all known-pixel gradients.
    p=x.detach().clone().requires_grad_(True)
    with autocast():zero=perceptual.per_image(p,x,m).mean()
    assert float(zero)==0
    p=(x+0.1*torch.randn_like(x)).detach().requires_grad_(True)
    with autocast():distance=perceptual.per_image(p,x,m).mean()
    distance.backward();assert ((1-m)*p.grad).abs().max()==0 and (m*p.grad).abs().max()>0
    # Independently reduce frozen feature maps and pixel differences in NumPy.
    with torch.no_grad(),autocast():
        y=x*(1-m);pred=net(y,y,m,torch.zeros(64,device='cuda')).float();pf=perceptual.encode(y+m*pred);tf=perceptual.encode(x)
    independent=np.stack([np.abs(a.float().cpu().numpy()-b.float().cpu().numpy()).mean((1,2,3))for a,b in zip(pf,tf)]).mean(0)
    np.testing.assert_allclose(independent,initial['feature_l1'].cpu().numpy(),rtol=2e-6,atol=2e-6)
    pp=pred.cpu().numpy();xx=x.cpu().numpy();mm=m.cpu().numpy();mse=(((pp-xx)**2)*mm).sum((1,2,3))/(3*mm.sum((1,2,3)))
    np.testing.assert_allclose(mse,initial['mse'].cpu().numpy(),rtol=2e-6,atol=2e-7)
    net.zero_grad(set_to_none=True);raw=predict(net,'regression',x[:8],m[:8],torch.zeros_like(x[:8]),1);assert ((raw-x[:8])*(1-m[:8])).abs().max()==0
    metric=lpips.LPIPS(net='squeeze',version='0.1').cuda().eval()
    for p in metric.parameters():p.requires_grad_(False)
    checks=check_metric_math(metric);assert checks['passed'];evaluate.metric=metric
    OUT.mkdir(parents=True);smoke=OUT/'dev_only_smoke';smoke.mkdir()
    evaluate(net,{'dev':cache['dev'][:4]},{'dev':bank['dev'][:4]},smoke,{'dev_pairs_per_image':4,'noise_seed':202610212,'nfe':[1]},0,30000,True)
    verify_saved(smoke,'regression',cache['dev'][:4],bank['dev'][:4])
    # Freeze calibration using training data only, before any optimizer update.
    cal={'training_images':64,'original_train_indices':ix['train'][:64],'mask_pair':0,'geometry':'1-arange64%2','initial_masked_mse':pixel,'initial_feature_l1':feature,'feature_scale':scale,'alpha_candidates':cfg['candidates'],'actual_coefficients':{tag:alpha*scale for tag,alpha in cfg['candidates'].items()},'vgg_weight_path':str(weight),'vgg_weight_url':VGG16_Weights.IMAGENET1K_V1.url,'vgg_weight_sha256':weight_hash,'frozen_feature_parameters':sum(p.numel()for p in perceptual.parameters()),'trainable_cnn_parameters':14986435,'optimizer_updates_before_calibration':0,'same_scale_for_both_training_seeds':True,'feature_input_clipped':False}
    save_json(OUT/'calibration.json',cal);save_json(OUT/'protocol.json',cfg)
    sources=['perceptual15_loss.py','train_perceptual15.py','verify_perceptual15.py','perceptual15_pipeline.py','preflight_perceptual15.py','perceptual15_protocol_v10.json','run_perceptual15_v10.sh','capacity15_models.py','pilot.py','train_matched.py','followup.py','continue30k.py','regression15_eval.py','compare_capacity.py','compare_regression15.py','official_test_v9.py','seed1_pipeline.py','matched_masks.py','matched_mask_protocol.json','data_manifest.json']
    old_files={}
    for seed in [0,1]:
        for kind,run in [('fm',cfg['fm_runs'][str(seed)]),('regression',cfg['mse_regression_runs'][str(seed)])]:
            base=ROOT/'runs'/run
            for filename in ['verification.json','checkpoint_manifest.json',f'seed{seed}/identity.json',f'seed{seed}/initial_weights.pt',f'seed{seed}/train.jsonl',f'seed{seed}/{kind}_step30000.pt',f'seed{seed}/{kind}_step30000_final_rows.json']:
                p=base/filename;old_files[str(p.relative_to(ROOT))]=digest(p)
    v9=ROOT/'runs/official_test_two_seeds_v9'
    for filename in ['protocol.json','checkpoint_freeze.json','verification.json','data_identity.json','comparison.json','RESULTS_zh.html']:
        p=v9/filename;old_files[str(p.relative_to(ROOT))]=digest(p)
    for seed in [0,1]:
        for kind in ['fm','regression']:
            p=v9/f'{kind}_seed{seed}/{kind}_step30000_final_rows.json';old_files[str(p.relative_to(ROOT))]=digest(p)
    frozen={'sources':{name:digest(ROOT/name)for name in sources},'calibration_sha256':digest(OUT/'calibration.json'),'old_frozen_artifacts':old_files,'post_test_supplement':True,'new_test_inference_before_dev_selection':False}
    save_json(OUT/'frozen_plan.json',frozen);source=OUT/'source';source.mkdir()
    for name in sources:(source/name).write_bytes((ROOT/name).read_bytes())
    result={'passed':True,'optimizer_updates':0,'cnn_initial_weights_equal_existing_models':True,'vgg_frozen':True,'finite_forward_backward_batch64_both_alphas':True,'identity_feature_distance_zero':True,'known_pixel_feature_gradients_zero':True,'numpy_feature_and_mse_reductions_checked':True,'dev_only_scoring_smoke_passed':True,'preflight_peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,'forward_backward_seconds':durations,'last_layer_gradient_norms':gradients,'calibration':cal,'source_freeze_sha256':digest(OUT/'frozen_plan.json')}
    save_json(OUT/'preflight.json',result);print('PREFLIGHT_PASSED',json.dumps(result),flush=True)
if __name__=='__main__':main()
