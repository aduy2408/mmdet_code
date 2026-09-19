import os

_base_ = ['./fcos_r50_set.py']

dataset_type = 'CocoDataset'
classes = ('varroa',)
set_root = os.environ.get('SET_ROOT', os.getcwd())
data_root = os.environ.get(
    'SET_VARROA_ROOT',
    os.path.abspath(os.path.join(
        set_root, '../mmdetection/mmdetection/data/varroa_coco')))
data = dict(
    train=dict(type=dataset_type, ann_file=data_root + '/annotations/train.json',
               img_prefix=data_root + '/images/train/', classes=classes),
    val=dict(type=dataset_type, ann_file=data_root + '/annotations/val.json',
             img_prefix=data_root + '/images/val/', classes=classes),
    test=dict(type=dataset_type, ann_file=data_root + '/annotations/test.json',
              img_prefix=data_root + '/images/test/', classes=classes))

model = dict(bbox_head=dict(num_classes=1))
