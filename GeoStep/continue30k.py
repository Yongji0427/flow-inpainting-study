"""Continue the immutable 10k models to 30k; fixed train/dev time-bin probes."""
import argparse
import copy
import hashlib
import json
import math
from pathlib import Path
import time

import lpips
import numpy as np
import torch
from torch import nn

from pilot import UNet, autocast, digest, evaluation_masks, masks, save_json, seed_all
from followup import evaluate, check_metric_math


def learning_rate(step, cfg):
    offset = step - cfg['initial_step']
    warm = cfg['learning_rate_warmup_updates']
    if offset <= warm:
        return cfg['learning_rate_start'] + offset / warm * (cfg['learning_rate_peak'] - cfg['learning_rate_start'])
    fraction = (offset - warm) / (cfg['final_step'] - cfg['initial_step'] - warm)
    return cfg['learning_rate_final'] + .5 * (cfg['learning_rate_peak'] - cfg['learning_rate_final']) * (1 + math.cos(math.pi * fraction))


def array_hash(value):
    return hashlib.sha256(value.detach().cpu().contiguous().numpy().tobytes()).hexdigest()


def probe_inputs(cache, cfg):
    gen = torch.Generator().manual_seed(cfg['train_probe_selection_seed'])
    indices = torch.randperm(len(cache['train']), generator=gen)[:cfg['train_probe_size']]
    splits = {'train_probe': cache['train'][indices].float() / 127.5 - 1,
              'dev': cache['dev'].float() / 127.5 - 1}
    bank = {}
    for split, images in splits.items():
        # The same independent CPU generator is rebuilt at every checkpoint.
        # Validation never consumes either training RNG stream.
        gen = torch.Generator().manual_seed(cfg['loss_probe_seed'])
        masksets = evaluation_masks(len(images), 'dev')
        bank[split] = {'images': images, 'masks': masksets, 'bins': []}
        for low, high in zip(cfg['loss_time_bins'][:-1], cfg['loss_time_bins'][1:]):
            t = low + (high - low) * torch.rand(len(images), generator=gen)
            eps = torch.randn(images.shape, generator=gen)
            bank[split]['bins'].append((low, high, t, eps))
        # Separate fixed endpoint probe, not a sampled training-time bin.
        bank[split]['endpoint_noise'] = torch.randn(images.shape, generator=gen)
    identity = {'train_indices': indices.tolist(), 'split_sizes': {k: len(v) for k, v in splits.items()},
                'images': {k: array_hash(v) for k, v in splits.items()},
                'masks': {s: {g: array_hash(m) for g, m in v['masks'].items()} for s, v in bank.items()},
                'times_and_noise': {s: [{'low': lo, 'high': hi, 't_sha256': array_hash(t),
                                       'noise_sha256': array_hash(e)} for lo, hi, t, e in v['bins']]
                                    for s, v in bank.items()}}
    return bank, identity


@torch.no_grad()
def loss_probe(net, kind, variant, step, bank):
    net.eval()
    rows = []
    for split, data in bank.items():
        for geometry, mask in data['masks'].items():
            probes = data['bins'] if kind == 'fm' else [(None, None, torch.zeros(len(mask)), None)]
            for low, high, times, eps in probes:
                for begin in range(0, len(mask), 32):
                    end = min(begin + 32, len(mask))
                    x = data['images'][begin:end].cuda()
                    m = mask[begin:end].cuda()
                    y = x * (1 - m)
                    t = times[begin:end].cuda()
                    if kind == 'fm':
                        e = eps[begin:end].cuda()
                        z = y + m * ((1 - t[:, None, None, None]) * e + t[:, None, None, None] * x)
                        target = x - e
                    else:
                        z, target = y, x
                    with autocast():
                        pred = net(z, y, m, t).float()
                    losses = ((pred - target).square() * m).sum((1, 2, 3)) / (3 * m.sum((1, 2, 3)))
                    assert torch.isfinite(losses).all()
                    for i, value in enumerate(losses.cpu().tolist()):
                        rows.append({'model': kind, 'variant': variant, 'step': step, 'split': split,
                                     'geometry': geometry, 'index': begin + i, 't_low': low, 't_high': high,
                                     't': float(times[begin+i]), 'loss': value})
    return rows


@torch.no_grad()
def endpoint_probe(net, kind, step, bank):
    # A fixed common-noise FM1/regression comparison on dev, without treating
    # the finite regressor as an oracle for the population conditional mean.
    data = bank['dev']
    outputs = {}
    for geometry, mask in data['masks'].items():
        chunks = []
        for begin in range(0, len(mask), 32):
            x = data['images'][begin:begin+32].cuda()
            m = mask[begin:begin+32].cuda()
            e = data['endpoint_noise'][begin:begin+32].cuda()
            y = x * (1-m)
            z = y+m*e if kind == 'fm' else y
            with autocast():
                v = net(z, y, m, torch.zeros(len(x), device='cuda')).float()
            pred = z+m*v if kind == 'fm' else y+m*v
            assert torch.isfinite(pred).all()
            assert ((pred-x)*(1-m)).abs().max() == 0
            chunks.append(pred.cpu().numpy())
        outputs[geometry] = np.concatenate(chunks)
    return outputs


def summarize_loss(rows):
    groups = {}
    for row in rows:
        key = (row['model'], row['variant'], row['step'], row['split'], row['geometry'], row['t_low'], row['t_high'])
        groups.setdefault(key, []).append(row['loss'])
    return [dict(zip(['model', 'variant', 'step', 'split', 'geometry', 't_low', 't_high'], key),
                 images=len(values), loss=float(np.mean(values))) for key, values in groups.items()]


def assess(root, out, cfg, kind, step, net, ema, bank, metric):
    clock = time.perf_counter()
    rows = []
    for variant, model in [('online', net), ('ema', ema)]:
        rows.extend(loss_probe(model, kind, variant, step, bank))
    stem = f'{kind}_step{step}'
    save_json(out / (stem + '_loss_rows.json'), rows)
    save_json(out / (stem + '_loss_summary.json'), summarize_loss(rows))
    evaluate(root, out, ema, kind, step, bank['dev']['images'], 'dev', metric, cfg)
    np.savez_compressed(out / (stem + '_endpoint_raw.npz'), **endpoint_probe(ema, kind, step, bank))
    print('ASSESS_COMPLETE', json.dumps({'model': kind, 'step': step, 'seconds': time.perf_counter()-clock}), flush=True)
    net.train()


def load_training(root, kind, cfg):
    path = root / 'runs/followup_seed0_v2' / f'{kind}_step10000.pt'
    checkpoint = torch.load(path, weights_only=False, map_location='cpu')
    assert checkpoint['kind'] == kind and checkpoint['step'] == cfg['initial_step']
    assert set(int(v['step']) for v in checkpoint['optimizer']['state'].values()) == {cfg['initial_step']}
    net = UNet().cuda()
    net.load_state_dict(checkpoint['model'])
    ema = copy.deepcopy(net).eval()
    ema.load_state_dict(checkpoint['ema'])
    opt = torch.optim.AdamW(net.parameters(), lr=cfg['learning_rate_start'], weight_decay=.0001)
    opt.load_state_dict(checkpoint['optimizer'])
    assert all(abs(g['lr'] - cfg['learning_rate_start']) < 1e-12 for g in opt.param_groups)
    dgen = torch.Generator(device='cuda')
    ngen = torch.Generator(device='cuda')
    dgen.set_state(checkpoint['data_generator_state'])
    ngen.set_state(checkpoint['noise_generator_state'])
    torch.set_rng_state(checkpoint['torch_rng'])
    torch.cuda.set_rng_state_all(checkpoint['cuda_rng'])
    return net, ema, opt, dgen, ngen


def preflight(root, cfg):
    assert abs(learning_rate(10000, cfg)-4e-5) < 1e-12
    assert abs(learning_rate(11000, cfg)-8e-5) < 1e-12
    assert abs(learning_rate(30000, cfg)-1e-5) < 1e-12
    result = {}
    cache = torch.load(root/'prepared.pt', weights_only=True)
    for kind in ['regression', 'fm']:
        net, ema, opt, dgen, ngen = load_training(root, kind, cfg)
        x = cache['train'][:8].cuda().float()/127.5-1
        m = masks(8, 'large', 7).cuda()
        y = x*(1-m)
        e = torch.randn(x.shape, device='cuda', generator=ngen)
        t = torch.rand(len(x), device='cuda', generator=ngen) if kind == 'fm' else torch.zeros(len(x), device='cuda')
        z = y+m*((1-t[:,None,None,None])*e+t[:,None,None,None]*x) if kind == 'fm' else y
        target = x-e if kind == 'fm' else x
        with autocast():
            loss = ((net(z,y,m,t).float()-target).square()*m).sum()/(3*m.sum())
        loss.backward()
        norm = nn.utils.clip_grad_norm_(net.parameters(), 1., error_if_nonfinite=True)
        assert torch.isfinite(loss)
        assert sum(p.numel() for p in net.parameters()) == 1747523
        result[kind] = {'finite_loss': float(loss), 'finite_gradient_norm': float(norm),
                        'optimizer_step': 10000, 'optimizer_updates_in_preflight': 0}
        del net, ema, opt
    print('PREFLIGHT', json.dumps(result), flush=True)
    return result


def run(root):
    cfg = json.loads((root/'continuation30k_protocol.json').read_text())
    out = root/'runs'/cfg['run']
    if out.exists():
        raise RuntimeError('Refusing to overwrite existing v3 run')
    old_identity = json.loads((root/'runs/followup_seed0_v2/identity.json').read_text())
    manifest = json.loads((root/'data_manifest.json').read_text())
    assert digest(root/'pilot.py') == old_identity['original_pilot_source_sha256']
    assert digest(root/'followup.py') == old_identity['followup_source_sha256']
    assert digest(root/'prepared.pt') == manifest['cache_sha256']
    assert digest('/data/LGFNet/manifests/splits.csv') == manifest['source_manifest_sha256']
    seed_all(0)
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    preflight_results = preflight(root, cfg)
    cache = torch.load(root/'prepared.pt', weights_only=True)
    bank, probe_identity = probe_inputs(cache, cfg)
    metric = lpips.LPIPS(net='squeeze', version='0.1').cuda().eval()
    for p in metric.parameters():
        p.requires_grad_(False)
    out.mkdir(parents=True)
    save_json(out/'protocol.json', cfg)
    save_json(out/'preflight.json', preflight_results)
    save_json(out/'metric_checks.json', check_metric_math(metric))
    save_json(out/'probe_identity.json', probe_identity)
    initial_hashes = {k: digest(root/f'runs/followup_seed0_v2/{k}_step10000.pt') for k in ['regression','fm']}
    save_json(out/'identity.json', {'source_sha256': digest(__file__),
              'protocol_sha256': digest(root/'continuation30k_protocol.json'),
              'pilot_sha256': digest(root/'pilot.py'), 'followup_sha256': digest(root/'followup.py'),
              'data_manifest_sha256': digest(root/'data_manifest.json'),
              'initial_checkpoint_sha256': initial_hashes,
              'probe_identity_sha256': digest(out/'probe_identity.json'),
              'torch_version': torch.__version__, 'gpu': torch.cuda.get_device_name()})
    train = cache['train'].cuda().float()/127.5-1
    mask_bank = torch.cat([masks(1024,g,300+i) for i,g in enumerate(['scattered','large'])]).cuda()
    save_json(out/'mask_identity.json', {'training_bank_sha256': array_hash(mask_bank),
              'training_bank_shape': list(mask_bank.shape),
              'areas': sorted(set(mask_bank.sum((1,2,3)).cpu().tolist())),
              'generator': 'unchanged pilot.masks, seeds 300/301, 1024 masks per geometry'})
    progress = {k: 10000 for k in ['regression', 'fm']}
    start = time.perf_counter()
    for kind in ['regression', 'fm']:
        net, ema, opt, dgen, ngen = load_training(root, kind, cfg)
        initial_rng = {'data': array_hash(dgen.get_state()), 'noise': array_hash(ngen.get_state())}
        assess(root,out,cfg,kind,10000,net,ema,bank,metric)
        stream = hashlib.sha256()
        train_seconds = 0.
        window = []
        torch.cuda.reset_peak_memory_stats()
        for step in range(cfg['initial_step']+1,cfg['final_step']+1):
            if time.perf_counter()-start > cfg['wall_seconds']:
                raise TimeoutError('v3 wall budget exceeded')
            clock = time.perf_counter()
            ix = torch.randint(len(train),(64,),device='cuda',generator=dgen)
            flip = torch.rand((64,1,1,1),device='cuda',generator=dgen)<.5
            mi = torch.randint(1024,(64,),device='cuda',generator=dgen)
            geom = (torch.arange(64,device='cuda')%2)*1024
            for value in [ix,flip,mi]:
                stream.update(value.cpu().numpy().tobytes())
            x = train[ix]
            x = torch.where(flip,x.flip(-1),x)
            m = mask_bank[mi+geom]
            y = x*(1-m)
            if kind == 'fm':
                e = torch.randn(x.shape,device='cuda',generator=ngen)
                t = torch.rand(64,device='cuda',generator=ngen)
                z = y+m*((1-t[:,None,None,None])*e+t[:,None,None,None]*x)
                target = x-e
            else:
                z,y,t,target = y,y,torch.zeros(64,device='cuda'),x
            lr = learning_rate(step,cfg)
            for group in opt.param_groups:
                group['lr'] = lr
            opt.zero_grad(set_to_none=True)
            with autocast():
                loss = ((net(z,y,m,t).float()-target).square()*m).sum()/(3*m.sum())
            if not torch.isfinite(loss):
                raise ValueError('nonfinite training loss')
            loss.backward()
            norm = nn.utils.clip_grad_norm_(net.parameters(),1.,error_if_nonfinite=True)
            opt.step()
            with torch.no_grad():
                for ep,np_ in zip(ema.parameters(),net.parameters()):
                    ep.lerp_(np_,1-cfg['ema_decay'])
            window.append(float(loss))
            train_seconds += time.perf_counter()-clock
            if step % 200 == 0:
                progress[kind] = step
                row = {'model':kind,'step':step,'learning_rate':lr,'loss_mean_last200':float(np.mean(window)),
                       'gradient_norm':float(norm),'training_seconds':train_seconds,
                       'batch_stream_sha256':stream.hexdigest(),
                       'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
                with (out/f'{kind}_train.jsonl').open('a') as f:
                    f.write(json.dumps(row,allow_nan=False)+'\n')
                save_json(out/'state.json',{'status':'training','progress':progress,**row})
                print('TRAIN_V3',json.dumps(row),flush=True)
                window = []
            if step in cfg['checkpoints']:
                assert set(int(v['step']) for v in opt.state.values()) == {step}
                torch.save({'model':net.state_dict(),'ema':ema.state_dict(),'optimizer':opt.state_dict(),
                            'step':step,'kind':kind,'config':cfg,'data_generator_state':dgen.get_state(),
                            'noise_generator_state':ngen.get_state(),'torch_rng':torch.get_rng_state(),
                            'cuda_rng':torch.cuda.get_rng_state_all(),'batch_stream_sha256':stream.hexdigest(),
                            'initial_rng_hashes':initial_rng},out/f'{kind}_step{step}.pt')
                assess(root,out,cfg,kind,step,net,ema,bank,metric)
        del net,ema,opt
    histories = {k:[json.loads(s) for s in (out/f'{k}_train.jsonl').read_text().splitlines()]
                 for k in ['regression','fm']}
    assert [r['batch_stream_sha256'] for r in histories['regression']] == [r['batch_stream_sha256'] for r in histories['fm']]
    assert initial_hashes == {k:digest(root/f'runs/followup_seed0_v2/{k}_step10000.pt') for k in initial_hashes}
    summary = {'status':'complete','new_optimizer_updates':40000,'final_step_each_model':30000,
               'wall_seconds':time.perf_counter()-start,'matched_training_data_stream':True,
               'network_parameters':1747523,'network_or_mask_changes':False,'official_test_access':False,
               'task_evaluation_split':'dev128','initial_checkpoint_hashes_unchanged':True,
               'training_seconds':{k:histories[k][-1]['training_seconds'] for k in histories},
               'limits':cfg['limitations']}
    save_json(out/'summary.json',summary)
    save_json(out/'state.json',{'status':'complete','progress':progress,'new_optimizer_updates':40000})
    print('CONTINUATION30K_COMPLETE',json.dumps(summary),flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=Path,required=True)
    parser.add_argument('--preflight',action='store_true')
    args = parser.parse_args()
    if args.preflight:
        cfg = json.loads((args.root/'continuation30k_protocol.json').read_text())
        preflight(args.root,cfg)
    else:
        try:
            run(args.root)
        except Exception as exc:
            out = args.root/'runs/continuation_seed0_30k_v3'
            if not (out/'summary.json').exists():
                save_json(out/'failure.json',{'type':type(exc).__name__,'message':str(exc)})
            raise
