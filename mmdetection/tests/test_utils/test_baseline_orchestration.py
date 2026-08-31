import json
import sys
from pathlib import Path


ROOT = Path(__file__).parents[3]
sys.path.insert(0, str(ROOT))

import evaluate_tinyperson_metrics as tiny_metrics  # noqa: E402
import run_two_server_baselines as two_server  # noqa: E402
import train_all_levir_baseline as levir  # noqa: E402
import train_all_tinyperson_baseline as tinyperson  # noqa: E402


def test_model_and_machine_manifests():
    expected = {'retinanet', 'cascade_rcnn', 'rtmdet'}
    tinyperson_expected = expected | {'atss', 'fcos', 'faster_rcnn'}
    assert expected <= set(levir.MODEL_CONFIGS)
    assert tinyperson_expected == set(tinyperson.MODEL_CONFIGS)
    assert set(two_server.JOBS) == {1, 2}
    assert {
        job for jobs in two_server.JOBS.values() for job in jobs
    } == {
        (dataset, model)
        for dataset in ('levir', 'tinyperson')
        for model in expected
    }


def test_tinyperson_pipeline_keeps_native_tile_scale():
    pipeline = tinyperson.pipeline(train=True)
    assert pipeline[0]['type'] == 'LoadTinyPersonImageFromFile'
    assert not any(transform['type'] == 'Resize' for transform in pipeline)
    assert tinyperson.dataset_config(
        Path('ann.json'), Path('images'), True, 0
    )['filter_cfg']['filter_empty_gt']


def test_validation_split_has_no_source_leakage(tmp_path):
    root = tmp_path / 'tiny_set'
    corner_path = root / tinyperson.TRAIN_ANN
    merged_path = root / tinyperson.MERGED_TRAIN_ANN
    corner_path.parent.mkdir(parents=True)
    merged_path.parent.mkdir(parents=True)
    sources = [
        {'id': index, 'file_name': f'{index}.jpg', 'width': 100, 'height': 100}
        for index in range(10)
    ]
    images = [
        {
            'id': index,
            'file_name': source['file_name'],
            'width': 64,
            'height': 64,
            'corner': [0, 0, 64, 64],
        }
        for index, source in enumerate(sources)
    ]
    annotations = [
        {
            'id': index,
            'image_id': index,
            'category_id': 1,
            'bbox': [1, 1, 5, 5],
            'area': 25,
            'iscrowd': 0,
        }
        for index in range(10)
    ]
    common = {'categories': [{'id': 1, 'name': 'person'}]}
    corner_path.write_text(json.dumps({
        **common, 'old_images': sources, 'images': images,
        'annotations': annotations,
    }))
    merged_path.write_text(json.dumps({
        **common, 'images': sources, 'annotations': annotations,
    }))
    paths = tinyperson.prepare_validation_split(root, tmp_path / 'out', 42, .2)
    train = json.loads(paths['train'].read_text())
    val = json.loads(paths['val'].read_text())
    train_names = {image['file_name'] for image in train['old_images']}
    val_names = {image['file_name'] for image in val['old_images']}
    assert len(val_names) == 2
    assert train_names.isdisjoint(val_names)


def test_tile_merge_translates_and_suppresses_duplicates(tmp_path):
    corner = {
        'old_images': [{'id': 10, 'file_name': 'a.jpg'}],
        'images': [
            {'id': 1, 'file_name': 'a.jpg', 'corner': [0, 0, 60, 60]},
            {'id': 2, 'file_name': 'a.jpg', 'corner': [5, 0, 65, 60]},
        ],
    }
    results = [
        {'image_id': 1, 'category_id': 1, 'bbox': [10, 10, 5, 5], 'score': .9},
        {'image_id': 2, 'category_id': 1, 'bbox': [5, 10, 5, 5], 'score': .8},
    ]
    corner_path = tmp_path / 'corner.json'
    result_path = tmp_path / 'result.json'
    output_path = tmp_path / 'merged.json'
    corner_path.write_text(json.dumps(corner))
    result_path.write_text(json.dumps(results))
    tiny_metrics.merge_detections(result_path, corner_path, output_path)
    merged = json.loads(output_path.read_text())
    assert len(merged) == 1
    assert merged[0]['image_id'] == 10
    assert merged[0]['bbox'] == [10, 10, 5, 5]
