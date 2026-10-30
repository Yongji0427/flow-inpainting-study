"""Artifact, update-count and independent NumPy ensemble-score verification."""
import argparse
import itertools
import json
from pathlib import Path
import numpy as np
import torch
from pilot import digest,evaluation_masks,save_json


def run(root):
    out=root/'runs/followup_seed0_v2';summary=json.loads((out/'summary.json').read_text())
    ident=json.loads((out/'identity.json').read_text())
    assert summary['status']=='complete' and summary['new_optimizer_updates']==16000
    assert ident['followup_source_sha256']==digest(root/'followup.py')
    assert ident['original_pilot_source_sha256']==digest(root/'pilot.py')
    assert ident['protocol_sha256']==digest(root/'followup_protocol.json')
    assert ident['data_manifest_sha256']==digest(root/'data_manifest.json')
    assert json.loads((out/'metric_checks.json').read_text())['passed']
    amendment=json.loads((out/'noise_amendment_identity.json').read_text())
    assert amendment['source_sha256']==digest(root/'evaluate_iid_noise.py')
    assert amendment['amendment_sha256']==digest(root/'NOISE_CONTROL_AMENDMENT_zh.md')
    manifest=json.loads((root/'data_manifest.json').read_text())
    assert manifest['source_manifest_sha256']==digest('/data/LGFNet/manifests/splits.csv')
    assert manifest['cache_sha256']==digest(root/'prepared.pt')
    histories={}
    for kind in ['regression','fm']:
        assert ident['initial_checkpoint_sha256'][kind]==digest(root/f'runs/pilot_seed0_v1/{kind}_step2000.pt')
        histories[kind]=[json.loads(s) for s in (out/f'{kind}_train.jsonl').read_text().splitlines()]
        assert [r['step'] for r in histories[kind]]==list(range(2200,10001,200))
        c=torch.load(out/f'{kind}_step10000.pt',weights_only=False,map_location='cpu')
        assert c['step']==10000 and c['kind']==kind
        assert set(int(v['step']) for v in c['optimizer']['state'].values())=={10000}
        assert all(torch.isfinite(v).all() for v in c['model'].values())
    assert [r['batch_stream_sha256'] for r in histories['regression']]==[r['batch_stream_sha256'] for r in histories['fm']]
    data=torch.load(root/'prepared.pt',weights_only=True)['eval'].float().numpy()/127.5-1
    masksets={k:v.numpy() for k,v in evaluation_masks(256,'eval').items()}
    checked=0;max_error=0.
    for step in [2000,10000]:
        for kind in ['regression','fm','iid_noise']:
            if kind=='iid_noise':
                stem=f'iid_noise_step{step}_eval';expected=[1]
            else:stem=f'{kind}_step{step}_eval';expected=[1] if kind=='regression' else [1,2,4,8,16,32]
            rows=json.loads((out/(stem+'_rows.json')).read_text())
            aggregate=json.loads((out/(stem+'_summary.json')).read_text())
            assert len(rows)==256*2*len(expected)
            examples=np.load(out/(stem+'_examples.npz'))
            for geom,mask in masksets.items():
                for nfe in expected:
                    rr=sorted([r for r in rows if r['geometry']==geom and r['nfe']==nfe],key=lambda r:r['index'])
                    assert [r['index'] for r in rr]==list(range(256))
                    a=next(r for r in aggregate if r['geometry']==geom and r['nfe']==nfe)
                    for metric in ['sample_mse','mean_prediction_mse','ensemble_variance','energy_score','lpips_truth','lpips_pairwise']:
                        np.testing.assert_allclose(np.mean([r[metric] for r in rr]),a[metric],atol=1e-10)
                    for r in rr:
                        assert abs(r['sample_mse']-r['mean_prediction_mse']-r['ensemble_variance'])<2e-7
                    pp=examples[geom if kind=='iid_noise' else f'{geom}_{nfe}']
                    assert pp.shape==(4,8,3,64,64)
                    assert pp.min()>=-1 and pp.max()<=1
                    assert np.max(np.abs((pp-data[None,:8])*(1-mask[None,:8])))==0
                    for index in range(8):
                        m=np.broadcast_to(mask[index]>0,(3,64,64))
                        pred=pp[:,index][:,m]/2
                        truth=data[index][m]/2
                        mse=((pred-truth[None])**2).mean(1).mean()
                        meanerr=((pred.mean(0)-truth)**2).mean()
                        var=((pred-pred.mean(0)[None])**2).mean()
                        es=np.linalg.norm(pred-truth[None],axis=1).mean()/np.sqrt(pred.shape[1])
                        es-=sum(np.linalg.norm(pred[i]-pred[j]) for i,j in itertools.combinations(range(4),2))/(12*np.sqrt(pred.shape[1]))
                        for key,value in [('sample_mse',mse),('mean_prediction_mse',meanerr),('ensemble_variance',var),('energy_score',es)]:
                            err=abs(float(value)-rr[index][key]);max_error=max(err,max_error)
                            assert err<3e-7,(step,kind,geom,nfe,index,key,err)
                        checked+=1
    result={'passed':True,'new_optimizer_updates':16000,'cumulative_main_optimizer_updates':20000,
       'training_data_streams_match':True,'independent_numpy_example_condition_checks':checked,
       'maximum_example_score_absolute_error':max_error,
       'checks':['frozen_source_protocol_and_data_identity','old_checkpoints_unchanged',
                 'optimizer_update_counts','finite_model_parameters','complete_metric_matrix',
                 'ensemble_mse_decomposition','independent_numpy_energy_scores',
                 'known_region_invariance','aggregate_means','corrected_iid_noise_artifacts'],
       'limits':['one seed','exploratory reused validation','four generated samples','no semantic calibration proof']}
    save_json(out/'verification.json',result);print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    run(p.parse_args().root)
