"""Independent artifact arithmetic and saved-example checks; no training."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
from pilot import evaluation_masks


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main(root):
    out=root/'runs/pilot_seed0_v1'
    summary=json.loads((out/'summary.json').read_text())
    ident=json.loads((out/'identity.json').read_text())
    manifest=json.loads((root/'data_manifest.json').read_text())
    assert summary['status']=='complete' and summary['main_optimizer_updates']==4000
    assert ident['source_sha256']==sha(root/'pilot.py')
    assert ident['protocol_sha256']==sha(root/'protocol.json')
    assert ident['data_manifest_sha256']==sha(root/'data_manifest.json')
    assert manifest['source_manifest_sha256']==sha('/data/LGFNet/manifests/splits.csv')
    assert manifest['cache_sha256']==sha(root/'prepared.pt')
    ids=manifest['ids']
    assert len(ids['train'])==2924 and len(ids['dev'])==128 and len(ids['eval'])==256
    assert not set(ids['train'])&set(ids['dev'])
    assert not set(ids['train'])&set(ids['eval'])
    assert not set(ids['dev'])&set(ids['eval'])
    ms=evaluation_masks(256,'eval')
    for g,m in ms.items():
        assert torch.all(m.sum((1,2,3))==1638)
        assert ident['eval_mask_sha256'][g]==hashlib.sha256(m.numpy().tobytes()).hexdigest()
    data=torch.load(root/'prepared.pt',weights_only=True)['eval'].float().numpy()/127.5-1
    example_checks=0
    for kind,nfes in [('regression',[1]),('fm',[1,2,4,8,16,32])]:
        rows=json.loads((out/f'{kind}_per_image.json').read_text())
        assert len(rows)==256*2*len(nfes)
        history=[json.loads(line) for line in (out/f'{kind}_train.jsonl').read_text().splitlines()]
        assert [r['step'] for r in history]==list(range(100,2001,100))
        assert all(np.isfinite(r['loss_mean_last100']) for r in history)
        artifact=torch.load(out/f'{kind}_evaluation.pt',weights_only=False,map_location='cpu')
        ckpt=torch.load(out/f'{kind}_step2000.pt',weights_only=False,map_location='cpu')
        assert ckpt['step']==2000 and ckpt['kind']==kind
        assert set(int(v['step']) for v in ckpt['optimizer']['state'].values())=={2000}
        for g in ['scattered','large']:
            for nfe in nfes:
                rr=sorted([r for r in rows if r['geometry']==g and r['nfe']==nfe],key=lambda r:r['index'])
                assert [r['index'] for r in rr]==list(range(256))
                for metric in ['mse','raw_mse','psnr','ssim_full']:
                    aggregate=next(s for s in summary['metrics'] if s['model']==kind and s['geometry']==g and s['nfe']==nfe)[metric]
                    np.testing.assert_allclose(np.mean([r[metric] for r in rr]),aggregate,rtol=0,atol=1e-10)
                pred=artifact['examples'][(g,nfe)].numpy()
                mask=ms[g][:8].numpy()
                errors=((pred-data[:8])/2)**2
                mse=(errors*mask).sum((1,2,3))/(3*mask.sum((1,2,3)))
                np.testing.assert_allclose(mse,[r['mse'] for r in rr[:8]],rtol=1e-6,atol=1e-7)
                assert np.max(np.abs((pred-data[:8])*(1-mask)))==0
                example_checks+=8
    frows=json.loads((out/'fm_per_image.json').read_text())
    gains=json.loads((out/'paired_gain.json').read_text())
    for metric in ['mse','raw_mse','psnr','ssim_full']:
        values={}
        for g in ['scattered','large']:
            lo=sorted((r for r in frows if r['geometry']==g and r['nfe']==1),key=lambda r:r['index'])
            hi=sorted((r for r in frows if r['geometry']==g and r['nfe']==16),key=lambda r:r['index'])
            values[g]=np.array([a[metric]-b[metric] for a,b in zip(lo,hi)])
            if metric in ['psnr','ssim_full']:values[g]*=-1
            np.testing.assert_allclose(values[g].mean(),gains[metric]['gain_'+g],atol=1e-10)
        np.testing.assert_allclose((values['large']-values['scattered']).mean(),gains[metric]['large_minus_scattered_gain'],atol=1e-10)
    result={'passed':True,'main_optimizer_updates':4000,'preflight_optimizer_updates':20,
            'saved_example_mse_and_known_region_checks':example_checks,
            'checks':['code_protocol_manifest_identity','source_manifest_unchanged','prepared_cache_identity',
                      'split_disjointness','equal_mask_area','complete_metric_matrix','training_step_counts',
                      'checkpoint_optimizer_step_counts','independent_numpy_example_mse',
                      'known_region_invariance','independent_aggregate_and_paired_gain_arithmetic'],
            'not_checked':['multiple seeds','new untouched test','LPIPS/FID','general semantic plausibility']}
    (out/'verification.json').write_text(json.dumps(result,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    main(p.parse_args().root)
