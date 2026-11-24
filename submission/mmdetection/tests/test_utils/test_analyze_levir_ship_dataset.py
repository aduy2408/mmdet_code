import importlib.util
from pathlib import Path

import numpy as np


SCRIPT = (
    Path(__file__).parents[2] / 'tools' / 'analysis_tools' /
    'analyze_levir_ship_dataset.py')
SPEC = importlib.util.spec_from_file_location('levir_analysis', SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_parse_filename():
    parsed = MODULE.parse_filename(
        'GF1_WFV3_E122.4_N37.3_20190805_L2A0004161911_10240_11264.png')
    assert parsed == {
        'filename_parse_error': False,
        'satellite': 'GF1',
        'sensor': 'WFV3',
        'longitude': 122.4,
        'latitude': 37.3,
        'capture_date': '2019-08-05',
        'product_id': 'L2A0004161911',
        'scene_id': 'GF1_WFV3_E122.4_N37.3_20190805_L2A0004161911',
        'tile_x': 10240,
        'tile_y': 11264,
    }
    assert MODULE.parse_filename('bad.png') == {'filename_parse_error': True}
    assert MODULE.parse_filename(
        'GF6_WFV_E138.3_N31.4_20200521_L1A1119999493-2_16384_17289.png'
    )['scene_id'] == 'GF6_WFV_E138.3_N31.4_20200521_L1A1119999493-2'


def test_read_yolo_and_geometry(tmp_path):
    annotation = tmp_path / 'sample.txt'
    annotation.write_text('0 0.5 0.25 0.2 0.1\nbroken\n')
    boxes = MODULE.read_yolo(annotation, width=100, height=200)
    assert boxes[0]['x'] == 40
    assert boxes[0]['y'] == 40
    assert boxes[0]['width'] == 20
    assert boxes[0]['height'] == 20
    assert boxes[1]['parse_error']
    assert MODULE.xywh_iou((0, 0, 10, 10), (5, 5, 10, 10)) == 25 / 175


def test_image_metrics_and_perceptual_hash():
    image = np.zeros((16, 16, 3), dtype=np.uint8)
    metrics = MODULE.image_metrics(image)
    assert metrics['black_ratio_le_0'] == 1
    assert metrics['white_ratio'] == 0
    assert metrics['entropy'] == 0
    assert MODULE.perceptual_hash(image) == MODULE.perceptual_hash(image.copy())


def test_annotation_matching():
    yolo = [{
        'line_number': 1,
        'parse_error': '',
        'class_id': 0,
        'x': 10.,
        'y': 20.,
        'width': 30.,
        'height': 40.,
    }]
    coco = [{'id': 1, 'category_id': 1, 'bbox': [10., 20., 30., 40.]}]
    assert MODULE._match_annotations('a.png', yolo, coco, .01) == []
    coco[0]['bbox'][0] = 12
    issues = MODULE._match_annotations('a.png', yolo, coco, .01)
    assert issues[0]['issue_type'] == 'bbox_coordinate_mismatch'


def test_robust_outliers():
    import pandas as pd

    flags = MODULE.robust_outlier_flags(pd.Series([1, 2, 2, 3, 100]))
    assert flags.tolist() == [False, False, False, False, True]
