# Compatibility base for the upstream SET configs.
# The model, optimizer, schedule, and runtime remain unchanged from upstream.
_base_ = ['./coco_detection.py']

dataset_type = 'CocoDataset'
data_root = 'data/aitod/'

# AI-TOD has eight object categories.  This is only the upstream compatibility
# definition; the dataset-specific configs below override it explicitly.
classes = ('airplane', 'bridge', 'ship', 'storage-tank', 'swimming-pool',
           'vehicle', 'person', 'windmill')

data = dict(
    train=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/trainval.json',
        img_prefix=data_root + 'images/',
        classes=classes),
    val=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/test.json',
        img_prefix=data_root + 'images/',
        classes=classes),
    test=dict(
        type=dataset_type,
        ann_file=data_root + 'annotations/test.json',
        img_prefix=data_root + 'images/',
        classes=classes))
