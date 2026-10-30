"""Small, bounded CNN inpainting pilot. The source dataset is always read-only."""
import argparse
import copy
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from scipy.ndimage import distance_transform_edt
import torch
from torch import nn
from torch.nn import functional as F


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + '.tmp')
    with tmp.open('w') as f:
        json.dump(obj, f, indent=2, allow_nan=False)
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def masks(count, geometry, seed):
    rng = np.random.default_rng(seed)
    out = np.zeros((count, 1, 64, 64), dtype=np.float32)
    for n in range(count):
        a = out[n, 0]
        if geometry == 'large':
            h, w = (39, 42) if rng.integers(2) else (42, 39)
            top, left = rng.integers(65-h), rng.integers(65-w)
            a[top:top+h, left:left+w] = 1
        else:
            # 16 connected patches, six with 103 and ten with 102 pixels.
            extra = set(rng.choice(16, 6, replace=False).tolist())
            for cell in range(16):
                patch = np.zeros((16, 16), dtype=np.float32)
                top, left = rng.integers(6), rng.integers(7)
                patch[top:top+10, left:left+10] = 1
                patch[top+10, left:left+2+(cell in extra)] = 1
                patch = np.rot90(patch, int(rng.integers(4)))
                r, c = divmod(cell, 4)
                a[16*r:16*r+16, 16*c:16*c+16] = patch
            a[:] = np.rot90(a.copy(), int(rng.integers(4)))
        assert a.sum() == 1638
    return torch.from_numpy(out)


def prepare(root, cfg):
    root.mkdir(parents=True, exist_ok=True)
    out = root / 'prepared.pt'
    if out.exists():
        print('PREPARED_EXISTS', flush=True)
        return
    records = list(csv.DictReader(open(cfg['source_manifest'])))
    train = sorted((r for r in records if r['split'] == 'train'), key=lambda r:r['image_id'])
    val = sorted((r for r in records if r['split'] == 'val'), key=lambda r:r['image_id'])
    assert len(train) == 2924 and len(val) == 730
    rng = np.random.default_rng(cfg['data_seed'])
    order = rng.permutation(len(val))
    dev = [val[i] for i in order[:cfg['dev_size']]]
    eva = [val[i] for i in order[cfg['dev_size']:cfg['dev_size']+cfg['eval_size']]]
    ids = {k:[r['image_id'] for r in v] for k,v in [('train',train),('dev',dev),('eval',eva)]}
    assert not set(ids['train']) & (set(ids['dev']) | set(ids['eval']))
    assert not set(ids['dev']) & set(ids['eval'])
    pixel_hashes = {}
    caches = {}
    for split, rows in [('train',train),('dev',dev),('eval',eva)]:
        arrays = []
        for r in rows:
            p = Path(cfg['image_root']) / (r['image_id']+'.jpg')
            assert digest(p) == r['file_sha256'], p
            with Image.open(p) as im:
                im = ImageOps.exif_transpose(im).convert('RGB')
                im = ImageOps.fit(im, (64,64), method=Image.Resampling.BICUBIC)
                a = np.asarray(im).copy()
            ph = hashlib.sha256(a.tobytes()).hexdigest()
            if ph in pixel_hashes and pixel_hashes[ph][0] != split:
                raise ValueError(f'Resized duplicate across splits: {r["image_id"]}')
            pixel_hashes[ph] = (split,r['image_id'])
            arrays.append(torch.from_numpy(a).permute(2,0,1))
        caches[split] = torch.stack(arrays)
        print('CACHED', split, len(rows), flush=True)
    torch.save(caches, out)
    save_json(root/'data_manifest.json', {'ids':ids,'source_manifest_sha256':digest(cfg['source_manifest']),
        'cache_sha256':digest(out),'resized_cross_split_exact_duplicates':0,
        'prior_near_duplicate_review':'Inherited LGFNet review; not rerun.',
        'final_test_images_opened':0})


class Block(nn.Module):
    def __init__(self, inc, outc, td=128):
        super().__init__()
        self.n1 = nn.GroupNorm(8, inc)
        self.c1 = nn.Conv2d(inc,outc,3,padding=1)
        self.t = nn.Linear(td,outc)
        self.n2 = nn.GroupNorm(8,outc)
        self.c2 = nn.Conv2d(outc,outc,3,padding=1)
        self.skip = nn.Conv2d(inc,outc,1) if inc != outc else nn.Identity()

    def forward(self,x,t):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.t(F.silu(t))[:,:,None,None]
        h = self.c2(F.silu(self.n2(h)))
        return (h+self.skip(x))/math.sqrt(2)


class UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('freq', torch.exp(torch.linspace(0,-math.log(10000),32)))
        self.temb = nn.Sequential(nn.Linear(64,128),nn.SiLU(),nn.Linear(128,128))
        self.inp = nn.Conv2d(7,32,3,padding=1)
        self.e0,self.e1,self.e2 = Block(32,32),Block(64,64),Block(96,96)
        self.d0,self.d1,self.d2 = nn.Conv2d(32,64,3,2,1),nn.Conv2d(64,96,3,2,1),nn.Conv2d(96,128,3,2,1)
        self.mid = nn.ModuleList([Block(128,128),Block(128,128)])
        self.u2,self.u1,self.u0 = nn.Conv2d(128,96,3,padding=1),nn.Conv2d(96,64,3,padding=1),nn.Conv2d(64,32,3,padding=1)
        self.b2,self.b1,self.b0 = Block(192,96),Block(128,64),Block(64,32)
        self.out = nn.Sequential(nn.GroupNorm(8,32),nn.SiLU(),nn.Conv2d(32,3,3,padding=1))
        nn.init.zeros_(self.out[-1].weight)
        nn.init.zeros_(self.out[-1].bias)

    def forward(self,z,y,m,t):
        angles = t.float()[:,None]*1000*self.freq[None]
        te = self.temb(torch.cat([angles.sin(),angles.cos()],dim=1))
        a = self.e0(self.inp(torch.cat([z,y,m],1)),te)
        b = self.e1(self.d0(a),te)
        c = self.e2(self.d1(b),te)
        h = self.d2(c)
        for block in self.mid:
            h = block(h,te)
        h = self.b2(torch.cat([self.u2(F.interpolate(h,scale_factor=2,mode='nearest')),c],1),te)
        h = self.b1(torch.cat([self.u1(F.interpolate(h,scale_factor=2,mode='nearest')),b],1),te)
        h = self.b0(torch.cat([self.u0(F.interpolate(h,scale_factor=2,mode='nearest')),a],1),te)
        return self.out(h)


def autocast():
    return torch.autocast('cuda',dtype=torch.bfloat16)


@torch.no_grad()
def predict(net,kind,x,m,eps,nfe=1,solver='euler'):
    y = x*(1-m)
    z = y+m*eps if kind == 'fm' else y
    t = torch.zeros(len(x),device=x.device)
    if kind == 'regression':
        with autocast():
            z = y+m*net(y,y,m,t).float()
    else:
        for k in range(nfe):
            t.fill_(k/nfe)
            with autocast():
                v = net(z,y,m,t).float()*m
            if solver == 'heun':
                t.fill_((k+1)/nfe)
                with autocast():
                    v2 = net(z+v/nfe,y,m,t).float()*m
                z = z+(v+v2)/(2*nfe)
            else:
                z = z+v/nfe
    known_error = float(((z-x)*(1-m)).abs().max())
    assert known_error == 0.0, known_error
    assert torch.isfinite(z).all()
    return z


def ssim_per_image(a,b):
    a,b=(a+1)/2,(b+1)/2
    coords=torch.arange(11,device=a.device,dtype=torch.float32)-5
    v=torch.exp(-coords.square()/(2*1.5**2));v=v/v.sum()
    w=(v[:,None]*v[None,:])[None,None].expand(3,1,-1,-1)
    def conv(t): return F.conv2d(t,w,groups=3)
    ma,mb=conv(a),conv(b)
    va,vb=conv(a*a)-ma*ma,conv(b*b)-mb*mb
    cov=conv(a*b)-ma*mb
    return (((2*ma*mb+0.01**2)*(2*cov+0.03**2))/((ma*ma+mb*mb+0.01**2)*(va+vb+0.03**2))).mean((1,2,3))


def evaluation_masks(n,split):
    offset = 700000 if split == 'dev' else 800000
    return {g:masks(n,g,offset+i) for i,g in enumerate(['scattered','large'])}


@torch.no_grad()
def evaluate(net,kind,images,masksets,nfes,batch=32,detail=False):
    net.eval()
    noise=torch.randn(images.shape,generator=torch.Generator().manual_seed(909099))
    rows=[]; examples={}; summaries=[]
    for geometry,mcpu in masksets.items():
        distances=np.stack([distance_transform_edt(v[0].numpy()) for v in mcpu])
        for nfe in nfes:
            torch.cuda.synchronize(); start=time.perf_counter()
            subset=[]
            for begin in range(0,len(images),batch):
                end=min(begin+batch,len(images))
                x=images[begin:end].cuda();m=mcpu[begin:end].cuda();eps=noise[begin:end].cuda()
                pred=predict(net,kind,x,m,eps,nfe)
                # Primary scores use the display-range output, also report raw MSE.
                clipped=pred.clamp(-1,1)
                sq=((clipped-x)/2).square()
                raw=((pred-x)/2).square()
                mse=(sq*m).sum((1,2,3))/(3*m.sum((1,2,3)))
                rawmse=(raw*m).sum((1,2,3))/(3*m.sum((1,2,3)))
                ssim=ssim_per_image(clipped,x)
                dist=torch.tensor(distances[begin:end],device='cuda')[:,None]
                for j in range(len(x)):
                    r={'index':begin+j,'geometry':geometry,'model':kind,'nfe':nfe,
                       'mse':float(mse[j]),'raw_mse':float(rawmse[j]),
                       'psnr':float(-10*torch.log10(mse[j].clamp_min(1e-12))),
                       'ssim_full':float(ssim[j]),'d90':float(np.percentile(distances[begin+j][distances[begin+j]>0],90)),
                       'out_of_range_fraction':float((((pred[j]<-1)|(pred[j]>1))*m[j]).sum()/(3*m[j].sum()))}
                    if detail:
                        for low,high,label in [(0,2,'near'),(2,5,'middle'),(5,100,'deep')]:
                            region=((dist[j]>low)&(dist[j]<=high))*m[j]
                            den=3*region.sum()
                            r['mse_'+label]=float((sq[j]*region).sum()/den) if den>0 else None
                    rows.append(r);subset.append(r)
                if begin==0:
                    examples[(geometry,nfe)]=clipped[:8].cpu()
            torch.cuda.synchronize(); elapsed=time.perf_counter()-start
            summaries.append({'geometry':geometry,'model':kind,'nfe':nfe,
                **{k:float(np.mean([r[k] for r in subset])) for k in ['mse','raw_mse','psnr','ssim_full','out_of_range_fraction']},
                'evaluation_seconds':elapsed,'images':len(subset)})
            print('EVAL',json.dumps(summaries[-1]),flush=True)
    return rows,summaries,examples


def preflight(root,cfg):
    seed_all(0)
    torch.set_num_threads(4)
    net=UNet().cuda()
    data=torch.load(root/'prepared.pt',weights_only=True)
    x=data['train'][:16].cuda().float()/127.5-1
    m=masks(8,'large',7).cuda()
    m=torch.cat([m,masks(8,'scattered',8).cuda()])
    e=torch.randn_like(x);t=torch.rand(16,device='cuda')
    y=x*(1-m);z=y+m*((1-t[:,None,None,None])*e+t[:,None,None,None]*x)
    assert torch.equal((z-x)*(1-m),torch.zeros_like(x))
    # Analytic path check: true conditional target integrates to x in one step.
    torch.testing.assert_close(y+m*e+m*(x-e),x,atol=5e-7,rtol=0)
    assert torch.all(m.sum((1,2,3))==cfg['missing_pixels'])
    opt=torch.optim.AdamW(net.parameters(),lr=cfg['learning_rate'])
    torch.cuda.synchronize();start=time.perf_counter();ls=[]
    for step in range(20):
        opt.zero_grad(set_to_none=True)
        with autocast():
            p=net(z,y,m,t)
            loss=((p.float()-(x-e)).square()*m).sum()/(3*m.sum())
        assert torch.isfinite(loss)
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in net.parameters() if p.grad is not None)
        torch.nn.utils.clip_grad_norm_(net.parameters(),1.0)
        opt.step();ls.append(float(loss))
    torch.cuda.synchronize();elapsed=time.perf_counter()-start
    predict(net,'fm',x,m,e,4)
    a=ssim_per_image(x,x)
    torch.testing.assert_close(a,torch.ones_like(a),atol=1e-5,rtol=0)
    save_json(root/'preflight.json',{'passed':True,'optimizer_updates':20,
        'parameters':sum(p.numel() for p in net.parameters()),'batch_size':16,
        'elapsed_seconds':elapsed,'loss_first':ls[0],'loss_last':ls[-1],
        'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30,
        'checks':['source_file_hashes','split_disjointness','resized_exact_duplicates',
                  'equal_missing_area','known_region_invariance','analytic_endpoint',
                  'finite_loss_gradients','ssim_identity'],
        'environment':{'torch':torch.__version__,'gpu':torch.cuda.get_device_name(),
                       'total_memory':torch.cuda.get_device_properties(0).total_memory}})
    print('PREFLIGHT_PASSED',flush=True)


def grid(root,images,masksets,examples,regexamples):
    n=8;cell=112;header=34;labels=['Original','Masked','Regression','FM 1','FM 4','FM 16','FM 32']
    canvas=Image.new('RGB',(cell*len(labels),header+2*n*cell+30),'white')
    draw=ImageDraw.Draw(canvas)
    for i,label in enumerate(labels):draw.text((i*cell+4,8),label,fill='black')
    for gi,g in enumerate(['scattered','large']):
        for i in range(n):
            x=images[i].float()/127.5-1;m=masksets[g][i]
            tiles=[x,x*(1-m),regexamples[(g,1)][i]]+[examples[(g,k)][i] for k in [1,4,16,32]]
            row=gi*n+i
            for j,a in enumerate(tiles):
                arr=((a.permute(1,2,0).clamp(-1,1)+1)*127.5).byte().numpy()
                tile=Image.fromarray(arr).resize((cell-8,cell-8),Image.Resampling.NEAREST)
                canvas.paste(tile,(j*cell+4,header+row*cell))
    draw.text((4,header+2*n*cell+4),'Top 8: scattered; bottom 8: large. Same images and fixed starting noise.',fill='black')
    canvas.save(root/'examples.png')


def paired_summary(rows):
    result={}
    for key in ['mse','raw_mse','psnr','ssim_full']:
        gains={}
        for g in ['scattered','large']:
            a={r['index']:r[key] for r in rows if r['geometry']==g and r['nfe']==1}
            b={r['index']:r[key] for r in rows if r['geometry']==g and r['nfe']==16}
            gains[g]=np.array([(a[i]-b[i]) if key in ['mse','raw_mse'] else (b[i]-a[i]) for i in sorted(a)])
        diff=gains['large']-gains['scattered']
        rng=np.random.default_rng(20261021)
        boot=diff[rng.integers(0,len(diff),size=(5000,len(diff)))].mean(1)
        result[key]={'gain_scattered':float(gains['scattered'].mean()),
                     'gain_large':float(gains['large'].mean()),
                     'large_minus_scattered_gain':float(diff.mean()),
                     'paired_image_bootstrap_95ci':np.percentile(boot,[2.5,97.5]).tolist()}
    return result


@torch.no_grad()
def reference_check(net,images,masksets):
    images=images[:64];eps=torch.randn(images.shape,generator=torch.Generator().manual_seed(909099))
    out=[]
    for g in ['scattered','large']:
        m=masksets[g][:64].cuda();x=images.cuda();e=eps.cuda()
        ref=predict(net,'fm',x,m,e,64)
        for solver,n in [('euler',1),('euler',4),('euler',16),('euler',32),('heun',8),('heun',16)]:
            p=predict(net,'fm',x,m,e,n,solver)
            out.append({'geometry':g,'solver':solver,'nfe':n*(2 if solver=='heun' else 1),
                        'mse_to_64nfe_reference':float((((p-ref)/2).square()*m).sum()/(3*m.sum())),
                        'clipped_mse_to_truth':float((((p.clamp(-1,1)-x)/2).square()*m).sum()/(3*m.sum()))})
    return out


def run(root,cfg):
    out=root/'runs'/'pilot_seed0_v1'
    if out.exists():raise RuntimeError('Refusing to overwrite existing pilot run')
    assert json.loads((root/'preflight.json').read_text())['passed']
    out.mkdir(parents=True)
    seed_all(cfg['seed']);torch.set_num_threads(4)
    torch.backends.cudnn.benchmark=True
    cache=torch.load(root/'prepared.pt',weights_only=True)
    train=(cache['train'].cuda().float()/127.5-1).contiguous()
    dev=cache['dev'].float()/127.5-1;eva=cache['eval'].float()/127.5-1
    bank=torch.cat([masks(cfg['mask_bank_per_geometry'],g,300+i) for i,g in enumerate(['scattered','large'])]).cuda()
    devm=evaluation_masks(len(dev),'dev');evalm=evaluation_masks(len(eva),'eval')
    init=UNet().state_dict()
    save_json(out/'protocol.json',cfg)
    save_json(out/'identity.json',{'source_sha256':digest(__file__),'protocol_sha256':digest(root/'protocol.json'),
             'data_manifest_sha256':digest(root/'data_manifest.json'),
             'mask_bank_sha256':hashlib.sha256(bank.cpu().numpy().tobytes()).hexdigest(),
             'eval_mask_sha256':{g:hashlib.sha256(m.numpy().tobytes()).hexdigest() for g,m in evalm.items()}})
    allsummary=[];regexamples=None;startall=time.perf_counter()
    for kind in ['regression','fm']:
        net=UNet().cuda();net.load_state_dict(init);ema=copy.deepcopy(net).eval()
        opt=torch.optim.AdamW(net.parameters(),lr=cfg['learning_rate'],weight_decay=cfg['weight_decay'])
        data_gen=torch.Generator(device='cuda').manual_seed(100)
        noise_gen=torch.Generator(device='cuda').manual_seed(200)
        log=out/(kind+'_train.jsonl');stepstart=time.perf_counter();window=[]
        torch.cuda.reset_peak_memory_stats()
        for step in range(1,cfg['steps']+1):
            if time.perf_counter()-startall>cfg['wall_seconds']:
                raise TimeoutError('Bounded main-process wall-clock cap reached')
            ix=torch.randint(len(train),(cfg['batch_size'],),device='cuda',generator=data_gen)
            x=train[ix]
            flip=torch.rand((len(x),1,1,1),device='cuda',generator=data_gen)<0.5
            x=torch.where(flip,x.flip(-1),x)
            mi=torch.randint(cfg['mask_bank_per_geometry'],(len(x),),device='cuda',generator=data_gen)
            geom=(torch.arange(len(x),device='cuda')%2)*cfg['mask_bank_per_geometry']
            m=bank[mi+geom];y=x*(1-m)
            if kind=='fm':
                eps=torch.randn(x.shape,device='cuda',generator=noise_gen)
                t=torch.rand(len(x),device='cuda',generator=noise_gen)
                z=y+m*((1-t[:,None,None,None])*eps+t[:,None,None,None]*x)
                target=x-eps
            else:
                z=y;t=torch.zeros(len(x),device='cuda');target=x
            opt.zero_grad(set_to_none=True)
            warm=min(1.,step/100)
            for group in opt.param_groups:
                group['lr']=cfg['learning_rate']*warm*(0.2+0.8*0.5*(1+math.cos(math.pi*step/cfg['steps'])))
            with autocast():
                pred=net(z,y,m,t)
                loss=((pred.float()-target).square()*m).sum()/(3*m.sum())
            if not torch.isfinite(loss):raise ValueError('Nonfinite loss')
            loss.backward()
            norm=torch.nn.utils.clip_grad_norm_(net.parameters(),1.0,error_if_nonfinite=True)
            opt.step()
            with torch.no_grad():
                for ep,p in zip(ema.parameters(),net.parameters()):ep.lerp_(p,1-cfg['ema_decay'])
            window.append(float(loss.detach()))
            if step%100==0:
                torch.cuda.synchronize()
                row={'model':kind,'step':step,'loss_mean_last100':float(np.mean(window)),
                     'gradient_norm':float(norm),'elapsed_seconds':time.perf_counter()-stepstart,
                     'peak_allocated_gib':torch.cuda.max_memory_allocated()/2**30}
                with log.open('a') as f:f.write(json.dumps(row)+'\n')
                save_json(out/'state.json',{'status':'training',**row})
                print('TRAIN',json.dumps(row),flush=True);window=[]
            if step in cfg['snapshots']:
                _,ss,_=evaluate(ema,kind,dev,devm,[1] if kind=='regression' else [1,4,16])
                save_json(out/f'{kind}_dev_step{step}.json',ss)
                torch.save({'model':net.state_dict(),'ema':ema.state_dict(),'optimizer':opt.state_dict(),
                            'step':step,'kind':kind,'config':cfg},out/f'{kind}_step{step}.pt')
        train_seconds=time.perf_counter()-stepstart
        rows,summary,examples=evaluate(ema,kind,eva,evalm,[1] if kind=='regression' else cfg['nfe'],detail=True)
        torch.save({'rows':rows,'summary':summary,'examples':examples},out/f'{kind}_evaluation.pt')
        save_json(out/f'{kind}_per_image.json',rows)
        save_json(out/f'{kind}_summary.json',{'train_and_dev_seconds':train_seconds,'metrics':summary})
        allsummary.extend(summary)
        if kind=='regression':regexamples=examples
        else:
            save_json(out/'paired_gain.json',paired_summary(rows))
            save_json(out/'reference_check.json',reference_check(ema,eva,evalm))
            grid(out,cache['eval'],evalm,examples,regexamples)
    # A no-learning masked-zero reference is scored on precisely the same images/masks.
    baseline=[]
    for g,m in evalm.items():
        mse=(((eva/2).square()*m).sum((1,2,3))/(3*m.sum((1,2,3)))).numpy()
        baseline.append({'geometry':g,'masked_zero_mse':float(mse.mean()),'masked_zero_psnr':float((-10*np.log10(mse)).mean())})
    save_json(out/'summary.json',{'status':'complete','main_optimizer_updates':2*cfg['steps'],
             'seed':0,'evaluation_images':len(eva),'wall_seconds':time.perf_counter()-startall,
             'metrics':allsummary,'zero_fill_baseline':baseline,
             'limitations':['one training seed','exploratory reused validation','Pet 64x64 only',
                           'one initial noise per image','no LPIPS or FID','no trained one-step generator',
                           'mask families differ in location/context distribution; no unique geometric causal attribution']})
    save_json(out/'state.json',{'status':'complete','main_optimizer_updates':2*cfg['steps']})
    print('PILOT_COMPLETE',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True)
    for flag in ['prepare','preflight','run']:p.add_argument('--'+flag,action='store_true')
    a=p.parse_args();cfg=json.loads((a.root/'protocol.json').read_text())
    if a.prepare:prepare(a.root,cfg)
    if a.preflight:preflight(a.root,cfg)
    if a.run:
        try:run(a.root,cfg)
        except Exception as exc:
            save_json(a.root/'runs/pilot_seed0_v1/failure.json',{'error':str(exc),'type':type(exc).__name__})
            raise
