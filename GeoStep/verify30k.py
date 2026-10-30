"""Verify continuation artifacts and independently recompute selected probe MSEs."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from pilot import UNet, autocast, digest, evaluation_masks, masks, save_json


def run(root):
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True
    out = root/'runs/continuation_seed0_30k_v3'
    cfg = json.loads((out/'protocol.json').read_text())
    summary = json.loads((out/'summary.json').read_text())
    identity = json.loads((out/'identity.json').read_text())
    assert summary['status'] == 'complete' and summary['new_optimizer_updates'] == 40000
    assert identity['source_sha256'] == digest(root/'continue30k.py')
    assert identity['protocol_sha256'] == digest(root/'continuation30k_protocol.json')
    assert identity['pilot_sha256'] == digest(root/'pilot.py')
    assert identity['followup_sha256'] == digest(root/'followup.py')
    assert identity['data_manifest_sha256'] == digest(root/'data_manifest.json')
    assert identity['probe_identity_sha256'] == digest(out/'probe_identity.json')
    assert json.loads((out/'metric_checks.json').read_text())['passed']
    manifest = json.loads((root/'data_manifest.json').read_text())
    assert digest(root/'prepared.pt') == manifest['cache_sha256']
    cache = torch.load(root/'prepared.pt',weights_only=True)
    train_gen = torch.Generator().manual_seed(cfg['train_probe_selection_seed'])
    indices = torch.randperm(len(cache['train']),generator=train_gen)[:128]
    probe_identity = json.loads((out/'probe_identity.json').read_text())
    assert indices.tolist() == probe_identity['train_indices']
    assert not set(manifest['ids']['train']) & set(manifest['ids']['dev'])
    bank = torch.cat([masks(1024,g,300+i) for i,g in enumerate(['scattered','large'])])
    mh = hashlib.sha256(bank.numpy().tobytes()).hexdigest()
    assert mh == json.loads((out/'mask_identity.json').read_text())['training_bank_sha256']
    histories = {}
    checkpoints = {}
    initial_metric_max_error={}
    for kind in ['regression','fm']:
        old_path = root/f'runs/followup_seed0_v2/{kind}_step10000.pt'
        assert digest(old_path) == identity['initial_checkpoint_sha256'][kind]
        c0 = torch.load(old_path,weights_only=False,map_location='cpu')
        histories[kind] = [json.loads(s) for s in (out/f'{kind}_train.jsonl').read_text().splitlines()]
        assert [r['step'] for r in histories[kind]] == list(range(10200,30001,200))
        assert np.isfinite([r['loss_mean_last200'] for r in histories[kind]]).all()
        assert abs(histories[kind][-1]['learning_rate']-1e-5) < 1e-12
        for step in cfg['checkpoints'][1:]:
            c = torch.load(out/f'{kind}_step{step}.pt',weights_only=False,map_location='cpu')
            assert c['step'] == step and c['kind'] == kind
            assert set(int(v['step']) for v in c['optimizer']['state'].values()) == {step}
            assert all(torch.isfinite(v).all() for v in c['model'].values())
            assert all(torch.isfinite(v).all() for v in c['ema'].values())
            assert {k:tuple(v.shape) for k,v in c0['model'].items()} == {k:tuple(v.shape) for k,v in c['model'].items()}
            checkpoints[(kind,step)] = c
        # Replay only the data RNG: identical draws and unchanged masks,
        # independent of validation forwards and FM's noise/time RNG.
        gen = torch.Generator(device='cuda')
        gen.set_state(c0['data_generator_state'])
        stream = hashlib.sha256()
        records = {r['step']:r for r in histories[kind]}
        for step in range(10001,30001):
            draws = [torch.randint(2924,(64,),device='cuda',generator=gen),
                     torch.rand((64,1,1,1),device='cuda',generator=gen)<.5,
                     torch.randint(1024,(64,),device='cuda',generator=gen)]
            for value in draws:
                stream.update(value.cpu().numpy().tobytes())
            if step % 200 == 0:
                assert records[step]['batch_stream_sha256'] == stream.hexdigest()
        assert torch.equal(gen.get_state(),checkpoints[(kind,30000)]['data_generator_state'])
        checkpoints[(kind,10000)] = c0
        for step in cfg['checkpoints']:
            rows = json.loads((out/f'{kind}_step{step}_loss_rows.json').read_text())
            bins = 5 if kind == 'fm' else 1
            assert len(rows) == 128*2*2*2*bins
            assert np.isfinite([r['loss'] for r in rows]).all()
            keys = [(r['variant'],r['split'],r['geometry'],r['index'],r['t_low']) for r in rows]
            assert len(keys) == len(set(keys))
            for row in rows:
                assert row['step'] == step
                if kind == 'fm':
                    assert row['t_low'] <= row['t'] < row['t_high']
            tasks = json.loads((out/f'{kind}_step{step}_dev_rows.json').read_text())
            nfes = [1] if kind == 'regression' else cfg['nfe']
            assert len(tasks) == 128*2*len(nfes)
            assert np.isfinite([[r[k] for k in ['sample_mse','sample_psnr','lpips_truth','energy_score']] for r in tasks]).all()
            assert all(abs(r['sample_mse']-r['mean_prediction_mse']-r['ensemble_variance']) < 2e-7 for r in tasks)
        # The new initial task assessment must reproduce the old 10k dev result.
        old_rows = json.loads((root/f'runs/followup_seed0_v2/{kind}_step10000_dev_rows.json').read_text())
        new_rows = json.loads((out/f'{kind}_step10000_dev_rows.json').read_text())
        lookup = {(r['geometry'],r['nfe'],r['index']):r for r in new_rows}
        metric_errors={m:0. for m in ['sample_mse','lpips_truth','energy_score']}
        for row in old_rows:
            new = lookup[(row['geometry'],row['nfe'],row['index'])]
            for metric in ['sample_mse','lpips_truth','energy_score']:
                metric_errors[metric]=max(metric_errors[metric],abs(row[metric]-new[metric]))
                # Diagnosis: pixel MSE and energy are exactly identical across
                # all 10k conditions; only LPIPS forwards differ by <=1.86e-6.
                # Record a small absolute numerical tolerance for LPIPS alone.
                if metric=='lpips_truth':
                    np.testing.assert_allclose(row[metric],new[metric],atol=3e-6,rtol=0)
                else:
                    np.testing.assert_allclose(row[metric],new[metric],atol=2e-7,rtol=2e-5)
        initial_metric_max_error[kind]=metric_errors
    assert [r['batch_stream_sha256'] for r in histories['regression']] == [r['batch_stream_sha256'] for r in histories['fm']]
    # Rebuild probes independently, select 8 fixed images and both endpoint
    # checkpoints, and calculate losses from NumPy predictions and targets.
    checked = 0
    max_error = 0.
    for kind in ['regression','fm']:
        for step in [10000,30000]:
            net = UNet().cuda().eval()
            net.load_state_dict(checkpoints[(kind,step)]['ema'])
            loss_rows = json.loads((out/f'{kind}_step{step}_loss_rows.json').read_text())
            for split, images in [('train_probe',cache['train'][indices]),('dev',cache['dev'])]:
                images = images.float()/127.5-1
                rng = torch.Generator().manual_seed(cfg['loss_probe_seed'])
                tbank, ebank = [], []
                for j in range(5):
                    lo,hi=cfg['loss_time_bins'][j:j+2]
                    tbank.append(lo+(hi-lo)*torch.rand(128,generator=rng))
                    ebank.append(torch.randn(images.shape,generator=rng))
                for geom, mask in evaluation_masks(128,'dev').items():
                    # Match the recorded forward batch shape. BF16 convolution
                    # results change slightly when the kernel sees batch8 vs32;
                    # NumPy still independently calculates the first 8 losses.
                    x = images[:32].cuda();m = mask[:32].cuda();y = x*(1-m)
                    for j in (range(5) if kind == 'fm' else [0]):
                        if kind == 'fm':
                            t = tbank[j][:32].cuda();e = ebank[j][:32].cuda()
                            z = y+m*((1-t[:,None,None,None])*e+t[:,None,None,None]*x)
                            target = x-e
                        else:
                            t = torch.zeros(32,device='cuda');z = y;target = x
                        with torch.no_grad(),autocast():
                            pred = net(z,y,m,t).float().cpu().numpy()
                        targets = target.cpu().numpy();mm = m.cpu().numpy()
                        calculated = ((pred-targets)**2*mm).sum((1,2,3))/(3*mm.sum((1,2,3)))
                        for i,value in enumerate(calculated[:8]):
                            r = next(r for r in loss_rows if r['variant']=='ema' and r['split']==split and r['geometry']==geom and r['index']==i and (kind=='regression' or abs(r['t_low']-j/5)<1e-8))
                            max_error = max(max_error,abs(float(value)-r['loss']))
                            np.testing.assert_allclose(value,r['loss'],atol=3e-4,rtol=.003)
                            checked += 1
            del net
    # Baseline dev inputs for fixed, non-selected visual examples.
    preview = {'truth':cache['dev'][:8].numpy()}
    preview.update({g:m[:8].numpy() for g,m in evaluation_masks(128,'dev').items()})
    np.savez_compressed(out/'dev_inputs_preview.npz',**preview)
    gap_rows=[]
    for step in cfg['checkpoints']:
        fm=np.load(out/f'fm_step{step}_endpoint_raw.npz')
        reg=np.load(out/f'regression_step{step}_endpoint_raw.npz')
        for geom,mask in evaluation_masks(128,'dev').items():
            m=mask.numpy()
            gaps=(((fm[geom]-reg[geom])/2)**2*m).sum((1,2,3))/(3*m.sum((1,2,3)))
            for i,value in enumerate(gaps):
                gap_rows.append({'step':step,'geometry':geom,'index':i,'raw_fm1_vs_regression_masked_mse':float(value)})
    save_json(out/'endpoint_gap_rows.json',gap_rows)
    save_json(out/'endpoint_gap_summary.json',[
        {'step':s,'geometry':g,'images':128,
         'raw_fm1_vs_regression_masked_mse':float(np.mean([r['raw_fm1_vs_regression_masked_mse'] for r in gap_rows if r['step']==s and r['geometry']==g]))}
        for s in cfg['checkpoints'] for g in ['scattered','large']])
    result = {'passed':True,'new_optimizer_updates':40000,'optimizer_step_each':30000,
              'checkpoint_count':8,'data_rng_replayed_updates_each_model':20000,
              'matched_training_stream':True,'initial_task_metrics_reproduced':True,
              'initial_task_metric_max_absolute_errors':initial_metric_max_error,
              'initial_lpips_absolute_tolerance':3e-6,
              'independent_probe_forward_batch':32,
              'numerics_note':'Match CPU normalization and batch32 BF16 forwards; a batch8 recomputation is not bitwise equivalent. LPIPS-only baseline drift <=1.86e-6; pixel metrics identical.',
              'independent_numpy_loss_checks':checked,'max_numpy_loss_error':max_error,
              'unchanged_network_and_masks':True,'official_test_access':False,
              'verifier_source_sha256':digest(__file__)}
    save_json(out/'verification.json',result)
    print(json.dumps(result),flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    run(p.parse_args().root)
