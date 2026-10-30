"""Frozen first official-test evaluation. No tuning, optimization, or checkpoint selection."""
import csv,hashlib,json,time
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps
import torch,lpips
from pilot import digest,save_json
from matched_masks import candidate_parameters,annotation,feature_maps,accepted,make_masks,counts
from train_matched import evaluate as fm_evaluate
from regression15_eval import evaluate as reg_evaluate
from followup import check_metric_math
from capacity15_models import LargeUNet
from compare_regression15 import paired
from compare_capacity import interval
from seed1_pipeline import render
ROOT=Path('/data/GeoStep')
OUT=ROOT/'runs/official_test_two_seeds_v9'
def read(p):return json.loads(Path(p).read_text())
def prepare(cfg):
    # checkpoint_freeze must already exist before any annotation() or JPEG access.
    freeze=read(OUT/'checkpoint_freeze.json');assert freeze['test_images_opened_before_freeze']==0
    for name,sha in freeze['source_freeze']['sources'].items():assert digest(ROOT/name)==sha
    pilotcfg=read(ROOT/'protocol.json');manifest=Path(pilotcfg['source_manifest'])
    assert digest(manifest)==cfg['source_manifest_expected_sha256']
    records=sorted([r for r in csv.DictReader(manifest.open())if r['split']=='test'],key=lambda r:r['image_id'])
    assert len(records)==cfg['expected_test_images']
    old=read(ROOT/'data_manifest.json');oldids=set(sum(old['ids'].values(),[]));assert not oldids.intersection(r['image_id']for r in records)
    # Repeat the exact resized-pixel duplicate audit before model evaluation.
    seen={}
    prepared=torch.load(ROOT/'prepared.pt',weights_only=True)
    assert digest(ROOT/'prepared.pt')==old['cache_sha256']
    for split,images in prepared.items():
        for image_id,x in zip(old['ids'][split],images):seen[hashlib.sha256(x.permute(1,2,0).contiguous().numpy().tobytes()).hexdigest()]=f'{split}/{image_id}'
    del prepared
    maskcfg=read(ROOT/'matched_mask_protocol.json');parameters=candidate_parameters(maskcfg);rng=np.random.default_rng(cfg['mask_selection_seed'])
    image_arrays=[];mask_arrays=[];metadata=[];excluded=[];source=hashlib.sha256();opened=0
    save_json(OUT/'state.json',{'status':'preparing','test_images_opened':0,'checkpoint_freeze_sha256':digest(OUT/'checkpoint_freeze.json')})
    for i,r in enumerate(records):
        image_id=r['image_id'];path=Path(pilotcfg['image_root'])/(image_id+'.jpg');assert digest(path)==r['file_sha256']
        with Image.open(path)as im:
            a=np.asarray(ImageOps.fit(ImageOps.exif_transpose(im).convert('RGB'),(64,64),method=Image.Resampling.BICUBIC)).copy()
        opened+=1;pixel_hash=hashlib.sha256(a.tobytes()).hexdigest()
        if pixel_hash in seen:
            excluded.append({'image_id':image_id,'reason':'exact_resized_duplicate','of':seen[pixel_hash]});continue
        seen[pixel_hash]='test/'+image_id
        trimap,head,src=annotation(ROOT,image_id);source.update(json.dumps({'id':image_id,**src},sort_keys=True).encode())
        features=feature_maps(trimap,head);eligible,big,small,totals,limits=accepted(features,parameters,maskcfg)
        if not len(eligible):excluded.append({'image_id':image_id,'reason':'no_v4_common_support'});continue
        eligible=rng.permutation(eligible);cost=(abs(big-small)[[0,2,3]][:,eligible]/np.maximum(limits[[0,2,3],None],1)).sum(0)
        pool=eligible[np.argsort(cost,kind='stable')[:32]];chosen=rng.choice(pool,size=4,replace=len(pool)<4)
        block=np.stack([make_masks(parameters[j])for j in chosen])
        direct=(block[:,:,None]*features[None,None]).sum((-1,-2)).astype(np.int64)
        np.testing.assert_array_equal(direct[:,0],big[:,chosen].T);np.testing.assert_array_equal(direct[:,1],small[:,chosen].T)
        assert np.all(abs(direct[:,0]-direct[:,1])<=limits)
        image_arrays.append(torch.from_numpy(a).permute(2,0,1));mask_arrays.append(torch.from_numpy(block.astype('bool')))
        metadata.append({'image_id':image_id,'official_index':i,'pixel_sha256':pixel_hash,'jpeg_sha256':r['file_sha256'],'candidate_indices':chosen.tolist(),'parameters':parameters[chosen].tolist(),'head_available':src['head_available'],'large_counts':direct[:,0].tolist(),'fragmented_counts':direct[:,1].tolist(),'calipers':limits.tolist()})
        if (i+1)%250==0:
            save_json(OUT/'state.json',{'status':'preparing','test_images_opened':opened,'retained':len(metadata),'official_images':len(records)});print('TEST_PREPARE',i+1,'retained',len(metadata),flush=True)
    audit={'passed':len(metadata)/len(records)>=cfg['minimum_retained_fraction'],'offered_test_images':len(records),'test_images_opened':opened,'retained':len(metadata),'retained_fraction':len(metadata)/len(records),'excluded':excluded,'annotation_source_sha256':source.hexdigest(),'no_model_outcomes_used_for_selection':True,'frozen_v4_mask_generator_sha256':digest(ROOT/'matched_masks.py')}
    save_json(OUT/'mask_audit.json',audit);save_json(OUT/'image_mask_metadata.json',metadata)
    assert audit['passed'],'Insufficient frozen-mask common support; stop without retuning'
    images=torch.stack(image_arrays);bank=torch.stack(mask_arrays);torch.save({'dev':images,'masks':bank},OUT/'prepared_test.pt')
    save_json(OUT/'data_identity.json',{'prepared_test_sha256':digest(OUT/'prepared_test.pt'),'metadata_sha256':digest(OUT/'image_mask_metadata.json'),'audit_sha256':digest(OUT/'mask_audit.json'),'source_manifest_sha256':digest(manifest),'evaluation_split':'official_test_common_support','images':len(images),'preprocessing':cfg['preprocessing']})
    return images,bank

def verify_saved(directory,kind,images,bank):
    rows=read(directory/f'{kind}_step30000_final_rows.json');assert len(rows)==len(images)*4*2
    assert len({(r['geometry'],r['image_index'],r['mask_pair'])for r in rows})==len(rows)
    assert np.isfinite([[r['lpips_truth'],r['sample_psnr'],r['sample_mse']]for r in rows]).all()
    examples=np.load(directory/f'{kind}_step30000_final_examples.npz');truth=images[:3].float().numpy()/127.5-1
    for gi,g in enumerate(['large','fragmented']):
        nfe=8 if kind=='fm'else 1;arr=examples[f'{g}_{nfe}'];m=bank[:3,0,gi,None].numpy();assert arr.shape==(4,3,3,64,64)
        assert np.max(abs((arr-truth[None])*(1-m[None])))<2e-7
        mse=((((arr-truth[None])/2)**2)*m[None]).sum((2,3,4))/(3*m.sum((1,2,3)))[None]
        psnr=(-10*np.log10(np.maximum(mse,1e-12))).mean(0)
        for i in range(3):
            row=next(r for r in rows if r['geometry']==g and r['image_index']==i and r['mask_pair']==0)
            np.testing.assert_allclose(mse[:,i].mean(),row['sample_mse'],rtol=2e-6,atol=2e-7);np.testing.assert_allclose(psnr[i],row['sample_psnr'],atol=2e-5,rtol=2e-6)
    return rows

def main():
    cfg=read(OUT/'protocol.json');assert cfg==read(ROOT/'official_test_protocol_v9.json')
    assert not (OUT/'state.json').exists(),'Refusing to rerun test access'
    freeze=read(OUT/'checkpoint_freeze.json')
    for item in freeze['checkpoints']:assert digest(item['path'])==item['sha256']
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=True;start=time.perf_counter()
    images,bank=prepare(cfg);metric=lpips.LPIPS(net='squeeze',version='0.1').cuda().eval()
    for p in metric.parameters():p.requires_grad_(False)
    checks=check_metric_math(metric);assert checks['passed'];save_json(OUT/'metric_checks.json',checks)
    weight_hashes={'lpips_trunk_sha256':digest('/data/geostep_cache/torch/hub/checkpoints/squeezenet1_1-b8a52dc0.pth'),'lpips_calibration_sha256':digest('/data/geostep_packages/lpips/weights/v0.1/squeeze.pth')}
    assert weight_hashes==read(ROOT/'runs/regression15_30k_v7/verification.json')['lpips_weight_hashes_equal_prior_frozen_metric']
    data={};fm_evaluate.metric=metric;reg_evaluate.metric=metric
    for item in freeze['checkpoints']:
        seed=item['seed'];kind=item['kind'];directory=OUT/f'{kind}_seed{seed}';directory.mkdir()
        save_json(OUT/'state.json',{'status':'evaluating','model':kind,'seed':seed,'test_images':len(images)})
        c=torch.load(item['path'],weights_only=False,map_location='cpu');assert c['step']==30000 and c['seed']==seed
        net=LargeUNet().cuda().eval();net.load_state_dict(c['ema']);del c
        evcfg={'dev_pairs_per_image':4,'noise_seed':cfg['noise_seed'],'nfe':[8 if kind=='fm'else 1]}
        function=fm_evaluate if kind=='fm'else reg_evaluate
        function(net,{'dev':images},{'dev':bank},directory,evcfg,seed,30000,True)
        data[seed,kind]=verify_saved(directory,kind,images,bank)
        assert digest(item['path'])==item['sha256'];del net
    display=[];sources={}
    for seed in cfg['seeds']:
        for geom in ['large','fragmented','overall']:
            for met in ['lpips_truth','sample_psnr']:
                row,source=paired(data[seed,'regression'],data[seed,'fm'],8,geom,met);gain=row['fm_improvement']
                display.append({'seed':seed,'geometry':geom,'metric':met,'regression_mean':row['regression_mean'],'fm_mean':row['fm_mean'],'fm_gain':gain['mean'],'ci95':gain['ci95']});sources[seed,geom,met]=source
    for geom in ['large','fragmented','overall']:
        for met in ['lpips_truth','sample_psnr']:
            a=sources[0,geom,met];b=sources[1,geom,met];assert [r['image_index']for r in a]==[r['image_index']for r in b]
            inter=interval([(x['fm_improvement']+y['fm_improvement'])/2 for x,y in zip(a,b)])
            display.append({'seed':'mean_of_2_seeds','geometry':geom,'metric':met,'regression_mean':float(np.mean([(x['regression']+y['regression'])/2 for x,y in zip(a,b)])),'fm_mean':float(np.mean([(x['fm']+y['fm'])/2 for x,y in zip(a,b)])),'fm_gain':inter['mean'],'ci95':inter['ci95']})
    save_json(OUT/'comparison.json',display)
    doc=render(display,'GeoStep：冻结配置后的正式测试')
    audit=read(OUT/'mask_audit.json');doc=doc.replace('</h1>',f'</h1><p>正式测试提供{audit["offered_test_images"]}张图；按预先冻结的重复图检查与掩码共同支持规则保留{len(images)}张，保留比例{audit["retained_fraction"]:.2%}。排除规则不读取模型结果；本报告结论针对该共同支持子集。</p>',1)
    doc=doc.replace('dev已经用于探索；正式测试结果单独报告。','本表来自预先冻结配置后的独立正式测试；没有根据测试结果选择种子、检查点或采样步数。')
    (OUT/'RESULTS_zh.html').write_text(doc)
    verification={'passed':True,'models_evaluated':4,'training_seeds':[0,1],'offered_official_test_images':audit['offered_test_images'],'retained_images':len(images),'conditions_per_model':len(images)*8,'initial_test_access_after_checkpoint_and_source_freeze':True,'frozen_checkpoint_hashes_unchanged':True,'saved_examples_known_pixels_and_numpy_mse_psnr_checked':True,'no_optimization_or_retuning':True,'lpips_weights':weight_hashes,'wall_seconds':time.perf_counter()-start,'report_sha256':digest(OUT/'RESULTS_zh.html'),'limitations':cfg['limitations']}
    save_json(OUT/'verification.json',verification);save_json(OUT/'state.json',{'status':'complete_verified',**verification});print('TEST_COMPLETE',json.dumps(verification),flush=True)
if __name__=='__main__':
    try:main()
    except Exception as e:
        save_json(OUT/'failure.json',{'type':type(e).__name__,'message':str(e)});raise
