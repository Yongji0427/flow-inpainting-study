"""Natural rectangular paired masks, matched before any model outcomes are read."""
import argparse
import hashlib
import json
from pathlib import Path
import xml.etree.ElementTree as ET

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import distance_transform_edt, label
import torch

from pilot import digest, save_json


def summed_area(a):
    return np.pad(a.astype(np.int64).cumsum(-2).cumsum(-1),((0,0),(1,0),(1,0)))


def rectangle_sum(integral, y, x, h, w):
    return integral[:,y+h,x+w]-integral[:,y,x+w]-integral[:,y+h,x]+integral[:,y,x]


def candidate_parameters(cfg):
    t,l,a,b=np.meshgrid(np.arange(1,33),np.arange(1,11),
                       cfg['horizontal_cut_choices'],cfg['vertical_cut_choices'],indexing='ij')
    t,l,a,b=[v.ravel() for v in [t,l,a,b]]
    py=np.where(t+a<=32,1,0);px=np.where(l+b<=32,1,0)
    ok=(t-py>=1)&(t-py+32<=63)&(l-px>=1)&(l-px+54<=63)
    p=np.stack([t[ok],l[ok],np.full(ok.sum(),31),np.full(ok.sum(),53),a[ok],b[ok],py[ok],px[ok]],1)
    transposed=p[:,[1,0,3,2,5,4,7,6]]
    p=np.concatenate([p,transposed]).astype(np.int64)
    # Match the actual missing pixels in center16, not just their count.
    yy,xx=np.mgrid[24:40,24:40];yy=yy.ravel()[None];xx=xx.ravel()[None]
    y,x,h,w,a,b,py,px=[v[:,None] for v in p.T]
    big=(yy>=y)&(yy<y+h)&(xx>=x)&(xx<x+w)
    small=(yy>=y-py)&(yy<y-py+h+1)&(xx>=x-px)&(xx<x-px+w+1)&(yy!=y-py+a)&(xx!=x-px+b)
    p=p[np.all(big==small,axis=1)]
    quadrants=[]
    for yy in [0,32]:
        for xx in [0,32]:
            q=np.zeros((64,64),dtype='uint8');q[yy:yy+32,xx:xx+32]=1;quadrants.append(q)
    large,small=counts(np.stack(quadrants),p)
    p=p[np.max(abs(large-small),axis=0)<=cfg['maximum_quadrant_missing_count_difference']]
    assert len(p)>0,'No spatially matched geometric candidates'
    return p


def counts(features,p):
    ii=summed_area(features)
    y,x,h,w,a,b,py,px=p.T
    big=rectangle_sum(ii,y,x,h,w)
    sy,sx=y-py,x-px
    small=(rectangle_sum(ii,sy,sx,a,b)+rectangle_sum(ii,sy,sx+b+1,a,w-b)+
           rectangle_sum(ii,sy+a+1,sx,h-a,b)+rectangle_sum(ii,sy+a+1,sx+b+1,h-a,w-b))
    big-=features[:,y,x]+features[:,y,x+1]+features[:,y,x+w-1]+features[:,y+h-1,x]+features[:,y+h-1,x+w-1]
    small-=features[:,sy,sx]+features[:,sy,sx+1]+features[:,sy,sx+w]+features[:,sy+h,sx]+features[:,sy+h,sx+w]
    return big,small


def make_masks(p):
    y,x,h,w,a,b,py,px=map(int,p)
    big=np.zeros((64,64),dtype='uint8');small=big.copy()
    big[y:y+h,x:x+w]=1
    sy,sx=y-py,x-px
    small[sy:sy+a,sx:sx+b]=1
    small[sy:sy+a,sx+b+1:sx+w+1]=1
    small[sy+a+1:sy+h+1,sx:sx+b]=1
    small[sy+a+1:sy+h+1,sx+b+1:sx+w+1]=1
    for yy,xx in [(y,x),(y,x+1),(y,x+w-1),(y+h-1,x),(y+h-1,x+w-1)]:big[yy,xx]=0
    for yy,xx in [(sy,sx),(sy,sx+1),(sy,sx+w),(sy+h,sx),(sy+h,sx+w)]:small[yy,xx]=0
    assert big.sum()==small.sum()==1638
    np.testing.assert_array_equal(big[24:40,24:40],small[24:40,24:40])
    assert label(big)[1]==1 and label(small)[1]==4
    return np.stack([big,small])


def orient(im,orientation):
    op={2:Image.Transpose.FLIP_LEFT_RIGHT,3:Image.Transpose.ROTATE_180,
        4:Image.Transpose.FLIP_TOP_BOTTOM,5:Image.Transpose.TRANSPOSE,
        6:Image.Transpose.ROTATE_270,7:Image.Transpose.TRANSVERSE,8:Image.Transpose.ROTATE_90}
    return im.transpose(op[orientation]) if orientation in op else im


def annotation(root,image_id):
    base=Path('/data/lgfnet_data/oxford-iiit-pet')
    png=base/'annotations/trimaps'/f'{image_id}.png'
    jpg=base/'images'/f'{image_id}.jpg'
    with Image.open(jpg) as image:
        size=image.size;orientation=image.getexif().get(274,1)
    with Image.open(png) as im:
        assert im.size==size,(image_id,im.size,size)
        trimap=np.asarray(ImageOps.fit(orient(im.copy(),orientation),(64,64),method=Image.Resampling.NEAREST)).copy()
    assert set(np.unique(trimap))<=set([1,2,3])
    xml=base/'annotations/xmls'/f'{image_id}.xml'
    head=np.zeros(size[::-1],dtype='uint8')
    if xml.exists():
        tree=ET.parse(xml)
        assert (int(tree.findtext('size/width')),int(tree.findtext('size/height')))==size
        for bb in tree.findall('object/bndbox'):
            xmin,ymin,xmax,ymax=[int(bb.findtext(k)) for k in ['xmin','ymin','xmax','ymax']]
            head[max(0,ymin-1):min(size[1],ymax),max(0,xmin-1):min(size[0],xmax)]=1
        head=np.asarray(ImageOps.fit(orient(Image.fromarray(head),orientation),(64,64),method=Image.Resampling.NEAREST)).copy()
    else:
        head=np.zeros((64,64),dtype='uint8')
    return trimap,head,{'orientation':orientation,'head_available':xml.exists(),
                       'trimap_sha256':digest(png),'head_xml_sha256':digest(xml) if xml.exists() else None}


def feature_maps(trimap,head):
    core=(trimap==1).astype('uint8');unknown=(trimap==3).astype('uint8')
    features=[core,unknown,(trimap!=2).astype('uint8'),head]
    for yy in [0,32]:
        for xx in [0,32]:
            q=np.zeros((64,64),dtype='uint8');q[yy:yy+32,xx:xx+32]=core[yy:yy+32,xx:xx+32]
            features.append(q)
    return np.stack(features)


def accepted(features,p,cfg):
    big,small=counts(features,p)
    totals=features.sum((1,2))
    calipers=[cfg[k] for k in ['core_foreground_caliper','possible_foreground_caliper','head_caliper']]
    def bound(c,n):return min(c['max_pixels'],np.floor(next(v for k,v in c.items() if k.startswith('max_fraction'))*n))
    limits=np.array([bound(calipers[0],totals[0]),cfg['uncertain_caliper_pixels'],
                     bound(calipers[1],totals[2]),bound(calipers[2],totals[3])]+
                    list(totals[4:]),dtype=np.int64)
    # Zero or very small foreground strata require exact count agreement.
    ok=np.all(np.abs(big-small)<=limits[:,None],axis=0)
    return np.flatnonzero(ok),big,small,totals,limits


def prepare(root,probe_only=False):
    cfg=json.loads((root/'matched_mask_protocol.json').read_text())
    directory=root/'matched_masks_v4'
    if (directory/'mask_bank.pt').exists():
        raise RuntimeError('Refusing to overwrite existing matched mask bank')
    directory.mkdir(exist_ok=True)
    manifest=json.loads((root/'data_manifest.json').read_text())
    assert digest(root/'prepared.pt')==manifest['cache_sha256']
    p=candidate_parameters(cfg)
    rng=np.random.default_rng(cfg['mask_selection_seed'])
    banks={};metadata={};audit={};failed=[];previews={}
    source_hash=hashlib.sha256()
    for split in ['train','dev']:
        ids=manifest['ids'][split]
        if probe_only:ids=ids[:128]
        pairs=cfg['train_pairs_per_image'] if split=='train' else cfg['dev_pairs_per_image']
        masks_out=[];meta=[];all_pairs=[];supports=[];head_missing=[];orientations=[]
        for i,image_id in enumerate(ids):
            trimap,head,source=annotation(root,image_id)
            source_hash.update(json.dumps({'id':image_id,**source},sort_keys=True).encode())
            features=feature_maps(trimap,head)
            eligible,big,small,totals,limits=accepted(features,p,cfg)
            supports.append(len(eligible));orientations.append(source['orientation'])
            if not source['head_available']:head_missing.append(image_id)
            if not len(eligible):
                failed.append({'split':split,'id':image_id,'totals':totals.tolist(),
                               'best_max_normalized_difference':float(np.min(np.max(abs(big-small)/np.maximum(limits[:,None],1),axis=0)))})
                continue
            eligible=rng.permutation(eligible)
            cost=(abs(big-small)[[0,2,3]][:,eligible]/np.maximum(limits[[0,2,3],None],1)).sum(0)
            pool=eligible[np.argsort(cost,kind='stable')[:32]]
            chosen=rng.choice(pool,size=pairs,replace=len(pool)<pairs)
            block=np.stack([make_masks(p[j]) for j in chosen])
            # Independent direct counting verifies integral-image count matching.
            direct=(block[:,:,None]*features[None,None]).sum((-1,-2)).astype(np.int64)
            np.testing.assert_array_equal(direct[:,0],big[:,chosen].T)
            np.testing.assert_array_equal(direct[:,1],small[:,chosen].T)
            masks_out.append(block)
            meta.append({'id':image_id,'original_index':i,'candidate_indices':chosen.tolist(),
                         'parameters':p[chosen].tolist(),'eligible_candidates':len(eligible),
                         'feature_totals':totals.tolist(),'count_calipers':limits.tolist(),
                         'large_counts':big[:,chosen].T.tolist(),'fragmented_counts':small[:,chosen].T.tolist(),
                         'head_available':source['head_available']})
            for j in range(pairs):
                row={'image_index':i,'pair':j,'core_difference':int(direct[j,0,0]-direct[j,1,0]),
                     'possible_foreground_difference':int(direct[j,0,2]-direct[j,1,2]),
                     'head_difference':int(direct[j,0,3]-direct[j,1,3])}
                all_pairs.append(row)
            if split=='dev' and i<8:
                previews[f'trimap_{i}']=trimap;previews[f'head_{i}']=head;previews[f'masks_{i}']=block
            if (i+1)%250==0:print('MATCHING',split,i+1,'minimum_support',min(supports),flush=True)
        support=np.array(supports)
        audit[split]={'images':len(ids),'retained_images':len(meta),'retained_fraction':len(meta)/len(ids),'candidate_support_min':int(support.min()),
                      'candidate_support_median':float(np.median(support)),
                      'zero_support':int((support==0).sum()),'head_annotation_missing':head_missing,
                      'nontrivial_exif_orientations':int(sum(o!=1 for o in orientations)),
                      'feature_names':['core_foreground','uncertain','possible_foreground','head_box',
                                       'core_top_left','core_top_right','core_bottom_left','core_bottom_right']}
        if masks_out:
            array=np.stack(masks_out)
            banks[split]=torch.from_numpy(array.astype('bool'))
            metadata[split]=meta
            means=array.mean((0,1))
            audit[split]['center8_coverage']=means[:,28:36,28:36].mean((1,2)).tolist()
            audit[split]['center16_coverage']=means[:,24:40,24:40].mean((1,2)).tolist()
            audit[split]['coverage_map_rmse']=float(np.sqrt(np.mean((means[0]-means[1])**2)))
            audit[split]['maximum_pixel_coverage_difference']=float(abs(means[0]-means[1]).max())
            audit[split]['matched_pairs']=len(all_pairs)
            audit[split]['max_abs_core_count_difference']=max(abs(r['core_difference']) for r in all_pairs)
            audit[split]['max_abs_head_count_difference']=max(abs(r['head_difference']) for r in all_pairs)
            audit[split]['mean_signed_core_difference']=float(np.mean([r['core_difference'] for r in all_pairs]))
            audit[split]['mean_signed_head_difference']=float(np.mean([r['head_difference'] for r in all_pairs]))
            audit[split]['mean_absolute_core_difference']=float(np.mean([abs(r['core_difference']) for r in all_pairs]))
            audit[split]['mean_absolute_head_difference']=float(np.mean([abs(r['head_difference']) for r in all_pairs]))
            audit[split]['max_abs_possible_foreground_difference']=max(abs(r['possible_foreground_difference']) for r in all_pairs)
            if split=='dev':
                np.savez_compressed(directory/'coverage_preview.npz',large=means[0],fragmented=means[1],**previews)
            # Geometry on fixed masks, without any image outcome inspection.
            geometry=[]
            for m in array[:min(128,len(array)),0]:
                rr=[]
                for g in range(2):
                    comp,n=label(m[g]);sizes=np.bincount(comp.ravel())[1:]
                    dist=distance_transform_edt(m[g])
                    yy,xx=np.where(m[g])
                    rr.append({'components':n,'largest_component':int(sizes.max()),
                               'd90':float(np.percentile(dist[m[g]>0],90)),
                               'centroid_y':float(yy.mean()),'centroid_x':float(xx.mean())})
                geometry.append(rr)
            audit[split]['geometry_probe']={'images':len(geometry),
                'component_counts':[sorted(set(r[g]['components'] for r in geometry)) for g in range(2)],
                'largest_component_ranges':[[min(r[g]['largest_component'] for r in geometry),max(r[g]['largest_component'] for r in geometry)] for g in range(2)],
                'mean_d90':[float(np.mean([r[g]['d90'] for r in geometry])) for g in range(2)],
                'max_centroid_difference':max(max(abs(r[0][key]-r[1][key]) for key in ['centroid_y','centroid_x']) for r in geometry)}
    retained_ok=all((v['images']-v['zero_support'])/v['images']>=cfg['minimum_retained_fraction'] for v in audit.values())
    result={'passed':retained_ok,'probe_only':probe_only,'candidates':len(p),'splits':audit,
            'no_support_images':failed,'official_test_images_opened':0,
            'annotation_source_sha256':source_hash.hexdigest(),
            'claim_boundary':'paired center and coarse position/foreground counts matched; pixelwise coverage need not be identical because visible fragmentation cuts are the intervention'}
    save_json(directory/('probe_audit.json' if probe_only else 'mask_audit.json'),result)
    print('MASK_AUDIT',json.dumps(result),flush=True)
    if not retained_ok:raise RuntimeError(f'Insufficient common support: {len(failed)} images unsupported; no training past audit gate')
    if not probe_only:
        torch.save(banks,directory/'mask_bank.pt')
        save_json(directory/'pair_metadata.json',metadata)
        save_json(directory/'supported_indices.json',{s:[r['original_index'] for r in mm] for s,mm in metadata.items()})
        save_json(directory/'identity.json',{'source_sha256':digest(__file__),
                  'protocol_sha256':digest(root/'matched_mask_protocol.json'),
                  'mask_bank_sha256':digest(directory/'mask_bank.pt'),
                  'pair_metadata_sha256':digest(directory/'pair_metadata.json'),
                  'supported_indices_sha256':digest(directory/'supported_indices.json'),
                  'data_manifest_sha256':digest(root/'data_manifest.json')})


if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--root',type=Path,required=True);p.add_argument('--probe-only',action='store_true')
    args=p.parse_args();prepare(args.root,args.probe_only)
