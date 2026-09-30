#!/usr/bin/env python3
"""DynamicVis adapter for the shared MMDetection YOLO-style Mosaic baseline."""
from __future__ import annotations
import argparse, json, os, random, shutil, sys
from pathlib import Path

sys.path[:0] = ['/marimo/DynamicVis', '/marimo/mmdet_code/mmdetection', '/marimo/mmdet_code']

PIPELINE = lambda n: [
    dict(type='Mosaic', img_scale=(n,n), pad_val=114.0, prob=1.0,
         pre_transform=[dict(type='LoadImageFromFile'), dict(type='LoadAnnotations', with_bbox=True)]),
    dict(type='RandomAffine', scaling_ratio_range=(0.5,1.5), max_rotate_degree=0.0,
         max_shear_degree=0.0, max_translate_ratio=0.1, border=(-n//2,-n//2),
         border_val=(114.0,114.0,114.0)),
    dict(type='YOLOXHSVRandomAug'),
    dict(type='RandomFlip', prob=0.5),
    dict(type='Resize', scale=(n,n), keep_ratio=False),
    dict(type='Pad', size=(n,n), pad_val=dict(img=(114,114,114))),
    dict(type='PackDetInputs'),
]
EVAL = lambda n: [dict(type='LoadImageFromFile'), dict(type='Resize', scale=(n,n), keep_ratio=False),
                  dict(type='LoadAnnotations', with_bbox=True), dict(type='PackDetInputs')]

def seed_all(seed):
    random.seed(seed); os.environ['PYTHONHASHSEED']=str(seed)
    import numpy as np, torch
    np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)

def coco_from_yolo(split_root: Path, out: Path, seed: int):
    from PIL import Image
    imgs=[]
    for p in sorted(split_root.rglob('*')):
        if p.suffix.lower() in {'.png','.jpg','.jpeg'}: imgs.append(p)
    rng=random.Random(seed); rng.shuffle(imgs)
    counts=(round(len(imgs)*2320/3896), round(len(imgs)*788/3896)) if len(imgs)==3896 else (round(len(imgs)*.6), round(len(imgs)*.2))
    train, val = imgs[:counts[0]], imgs[counts[0]:counts[0]+counts[1]]; test=imgs[counts[0]+counts[1]:]
    ann=out/'annotations'; ann.mkdir(parents=True,exist_ok=True)
    for name, selected in [('train',train),('val',val),('test',test)]:
        records=[]; annotations=[]; aid=1
        for iid,p in enumerate(selected,1):
            with Image.open(p) as im: w,h=im.size
            records.append(dict(id=iid,file_name=str(p),width=w,height=h))
            label=split_root.parent/'All Annotations'/(p.stem+'.txt')
            if label.is_file():
                for line in label.read_text().splitlines():
                    x=line.split()
                    if len(x)!=5: continue
                    _,xc,yc,bw,bh=map(float,x); boxw=bw*w; boxh=bh*h; x0=xc*w-boxw/2; y0=yc*h-boxh/2
                    annotations.append(dict(id=aid,image_id=iid,category_id=1,bbox=[x0,y0,boxw,boxh],area=boxw*boxh,iscrowd=0)); aid+=1
        (ann/f'{name}.json').write_text(json.dumps(dict(images=records,annotations=annotations,categories=[dict(id=1,name='ship')]),indent=2))
    return out

def make_cfg(args, dataset_root: Path, out: Path):
    from mmengine.config import Config
    base='/marimo/DynamicVis/configs_DynamicVis/Levir-Ship/dynamicvis_b_levirship_mamba.py'
    cfg=Config.fromfile(base)
    cfg.custom_imports=dict(imports=['projects.set','mmdet.datasets.transforms','dynamicvis'],allow_failed_imports=False)
    cfg.data_root=str(dataset_root); cfg.work_dir=str(out); cfg.img_size=args.imgsz; cfg.crop_size=(args.imgsz,args.imgsz)
    cfg.pretrained_ckpt='/marimo/DynamicVis/checkpoints/pretrain_dynamicvis_b_bf16_mamba_epoch_200.pth'
    meta=dict(classes=('person',) if args.dataset=='tinyperson' else ('ship',))
    cfg.train_dataloader=dict(batch_size=8,num_workers=8,persistent_workers=True,sampler=dict(type='DefaultSampler',shuffle=True),
        dataset=dict(type='CocoDataset',data_root=str(dataset_root),ann_file='annotations/train.json',data_prefix=dict(img=''),metainfo=meta,pipeline=PIPELINE(args.imgsz)))
    cfg.val_dataloader=dict(batch_size=8,num_workers=8,persistent_workers=True,drop_last=False,sampler=dict(type='DefaultSampler',shuffle=False),
        dataset=dict(type='CocoDataset',data_root=str(dataset_root),ann_file='annotations/val.json',data_prefix=dict(img=''),metainfo=meta,test_mode=True,pipeline=EVAL(args.imgsz)))
    cfg.test_dataloader=cfg.val_dataloader
    cfg.val_evaluator=dict(type='CocoMetric',metric=['bbox'],iou_thrs=[0.5],ann_file=str(dataset_root/'annotations/val.json'))
    cfg.test_evaluator=dict(type='CocoMetric',metric=['bbox'],iou_thrs=[0.5],ann_file=str(dataset_root/'annotations/test.json'))
    cfg.train_cfg=dict(type='EpochBasedTrainLoop',max_epochs=100,val_interval=1)
    cfg.optim_wrapper=dict(type='AmpOptimWrapper',loss_scale='dynamic',optimizer=dict(type='MuSGD',lr=0.01,momentum=0.9,nesterov=True,weight_decay=0.0005,muon=0.2,sgd=1.0),paramwise_cfg=dict(bias_lr_mult=1.0,bias_decay_mult=0.0),clip_grad=dict(max_norm=35,norm_type=2))
    cfg.param_scheduler=[dict(type='LinearLR',start_factor=0.001,by_epoch=True,begin=0,end=3),dict(type='LinearLR',start_factor=1.0,end_factor=0.01,by_epoch=True,begin=3,end=100)]
    cfg.randomness=dict(seed=42,deterministic=True)
    cfg.vis_backends=[dict(type='LocalVisBackend')]
    cfg.visualizer=dict(type='DetLocalVisualizer',vis_backends=cfg.vis_backends,name='visualizer',line_width=2)
    cfg.default_hooks.checkpoint=dict(type='CheckpointHook',interval=1,by_epoch=True,max_keep_ckpts=1,save_last=True,save_best='coco/bbox_mAP',rule='greater')
    cfg.custom_hooks=[dict(type='EarlyStoppingHook',monitor='coco/bbox_mAP',rule='greater',patience=15,min_delta=0.001)]
    cfg.model.bbox_head.num_classes=1; cfg.model.test_cfg.nms=dict(type='nms',iou_threshold=0.5)
    return cfg

def main():
    p=argparse.ArgumentParser(); p.add_argument('--dataset',choices=['tinyperson','levirship'],required=True); p.add_argument('--data-root',type=Path,required=True); p.add_argument('--work-dir',type=Path,required=True); p.add_argument('--model-yaml',required=True); p.add_argument('--epochs',type=int,default=100); p.add_argument('--patience',type=int,default=15); p.add_argument('--workers',type=int,default=8); p.add_argument('--seed',type=int,default=42); p.add_argument('--split-seed',type=int,default=42); p.add_argument('--nms-iou',type=float,default=.5); p.add_argument('--hf-repo-id',required=True); p.add_argument('--imgsz',type=int,default=640); p.add_argument('--prepare-only',action='store_true'); args=p.parse_args(); seed_all(args.seed)
    out=args.work_dir; out.mkdir(parents=True,exist_ok=True)
    if args.dataset=='tinyperson':
        root=out/'dataset'; root.mkdir(exist_ok=True); (root/'annotations').mkdir(exist_ok=True)
        src=Path('/marimo/TinyPerson/annotations'); shutil.copy2(src/'tiny_set_train.json',root/'annotations/train.json'); shutil.copy2(src/'tiny_set_test_with_dense.json',root/'annotations/val.json'); shutil.copy2(src/'tiny_set_test_with_dense.json',root/'annotations/test.json'); data=root
        # rewrite image file paths to absolute paths used by the mounted dataset
        for name in ['train','val','test']:
            q=root/'annotations'/f'{name}.json'; d=json.loads(q.read_text()); prefix='erase_with_uncertain_dataset/train/' if name=='train' else 'erase_with_uncertain_dataset/test/'
            for im in d['images']: im['file_name']=str(Path('/marimo/TinyPerson')/prefix/im['file_name'].replace('labeled_images/','labeled_images/'))
            q.write_text(json.dumps(d))
    else: data=coco_from_yolo(Path('/marimo/LevirShip/LevirShipData/All Images'),out/'dataset',args.split_seed)
    cfg=make_cfg(args,data,out); cfg.dump(str(out/'resolved_config.py')); (out/'experiment_manifest.json').write_text(json.dumps(dict(dataset=args.dataset,seed=42,split_seed=42,epochs=100,patience=15,batch_size=8,workers=8,optimizer='MuSGD',augmentation='shared_yolo_train_pipeline Mosaic p=1.0 + RandomAffine + HSV + flip + Resize + Pad',nms_iou=.5,pretrained=cfg.pretrained_ckpt),indent=2))
    if args.prepare_only: return
    from mmengine.runner import Runner
    runner=Runner.from_cfg(cfg); runner.train(); runner.test()
    from huggingface_hub import HfApi
    token=os.environ.get('HF_TOKEN'); api=HfApi(token=token); api.create_repo(repo_id=args.hf_repo_id,repo_type='dataset',exist_ok=True); api.upload_folder(folder_path=str(out),path_in_repo=args.dataset,repo_id=args.hf_repo_id,repo_type='dataset')

if __name__=='__main__': main()
