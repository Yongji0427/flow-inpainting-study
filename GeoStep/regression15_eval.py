"""Direct-regression evaluation with frozen v4 conditions and v2 metric math."""
import json
import numpy as np
import torch
from pilot import predict,save_json
from followup import scores
from continue30k import loss_probe,summarize_loss

@torch.no_grad()
def evaluate(net,cache,maskbank,out,cfg,seed,step,all_pairs):
    net.eval()
    original=cache['dev'].float()/127.5-1
    pairs=cfg['dev_pairs_per_image'] if all_pairs else 1
    x=original[:,None].expand(-1,pairs,-1,-1,-1).reshape(-1,3,64,64)
    rows=[];summaries=[];examples={}
    for gi,geometry in enumerate(['large','fragmented']):
        mm=maskbank['dev'][:,:pairs,gi,None].reshape(-1,1,64,64).float()
        for nfe in cfg['nfe']:
            group=[]
            for begin in range(0,len(x),32):
                end=min(begin+32,len(x));b=end-begin
                xb=x[begin:end].cuda();m=mm[begin:end].cuda()
                # Four identical copies use the same frozen metric code as FM.
                # These are one deterministic output, not independent samples.
                one=predict(net,'regression',xb,m,torch.zeros_like(xb),1)
                raw=one[None].expand(4,-1,-1,-1,-1)
                pred=raw.clamp(-1,1)
                values=scores(pred,raw,xb,m,evaluate.metric)
                for j in range(b):
                    condition=begin+j
                    row={'model':'regression','seed':seed,'step':step,'geometry':geometry,'nfe':nfe,
                         'condition_index':condition,'image_index':condition//pairs,'mask_pair':condition%pairs,
                         **{k:float(v[j]) for k,v in values.items()}}
                    rows.append(row);group.append(row)
                if begin==0:
                    examples[f'{geometry}_{nfe}']=pred[:,[0,pairs,2*pairs]].cpu().numpy()
            summary={'model':'regression','seed':seed,'step':step,'geometry':geometry,'nfe':nfe,
                     'images':len(original),'mask_pairs':pairs,'conditions':len(x),
                     **{k:float(np.mean([r[k] for r in group])) for k in values}}
            summaries.append(summary)
            print('TASK',json.dumps(summary),flush=True)
    stage='final' if all_pairs else 'curve'
    stem=f'regression_step{step}_{stage}'
    save_json(out/f'{stem}_rows.json',rows)
    save_json(out/f'{stem}_summary.json',summaries)
    if all_pairs:np.savez_compressed(out/f'{stem}_examples.npz',**examples)
    return summaries


def assess(net,ema,cache,maskbank,out,cfg,seed,step,probes):
    rows=[]
    for variant,model in [('online',net),('ema',ema)]:
        rows.extend(loss_probe(model,'regression',variant,step,probes))
    for row in rows:row['seed']=seed
    save_json(out/f'regression_step{step}_loss_rows.json',rows)
    save_json(out/f'regression_step{step}_loss_summary.json',summarize_loss(rows))
    evaluate(ema,cache,maskbank,out,cfg,seed,step,False)
    net.train()

