import os

_base_ = ['./coco_detection.py']

dataset_type = 'CocoDataset'
classes = ('ship',)
set_root = os.environ.get('SET_ROOT', os.getcwd())
data_root = os.environ.get(
    'SET_LEVIR_ROOT',
    os.path.abspath(os.path.join(set_root, 'data/levir_ship')))
data = dict(
    train=dict(type=dataset_type, ann_file=data_root + '/annotations/train.json',
               img_prefix=data_root + '/images/', classes=classes),
    val=dict(type=dataset_type, ann_file=data_root + '/annotations/val.json',
             img_prefix=data_root + '/images/', classes=classes),
    test=dict(type=dataset_type, ann_file=data_root + '/annotations/test.json',
              img_prefix=data_root + '/images/', classes=classes))
