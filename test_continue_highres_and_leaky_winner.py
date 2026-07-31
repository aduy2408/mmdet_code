import random
from pathlib import Path
from unittest.mock import Mock

import continue_highres_and_leaky_winner as continuation
from continue_highres_and_leaky_winner import SPLIT_TARGETS, select_exact


def test_exact_count_partition():
    annotation_counts = [0] * 2156 + [1] * 1000 + [2] + [3] * 739
    records = [
        {
            'image': {'id': index, 'file_name': f'{index}.png'},
            'annotations': [{}] * count,
        }
        for index, count in enumerate(annotation_counts)
    ]
    rng = random.Random(42)
    test, remaining = select_exact(records, *SPLIT_TARGETS['test'], rng)
    val, train = select_exact(remaining, *SPLIT_TARGETS['val'], rng)
    for split, selected in (
            ('train', train), ('val', val), ('test', test)):
        assert (
            len(selected),
            sum(len(record['annotations']) for record in selected),
        ) == SPLIT_TARGETS[split]


def test_existing_reuses_completed_checkpoint_test(tmp_path: Path, monkeypatch):
    sha = 'a' * 40
    run = tmp_path / sha / '1376' / 'raw'
    checkpoint = run / 'best.pth'
    checkpoint.parent.mkdir(parents=True)
    checkpoint.touch()
    test_dir = run / 'checkpoint_tests' / 'best'
    test_dir.mkdir(parents=True)
    (test_dir / 'run_summary.json').write_text(
        '{"duration_seconds": 2.5, "metrics": '
        '{"coco/bbox_mAP": 0.31}}',
        encoding='utf-8',
    )
    monkeypatch.setattr(continuation.repair, 'DEFAULT_VARIANTS', ('raw',))
    monkeypatch.setattr(
        continuation.repair, 'checkpoints', lambda _config: [checkpoint])
    test_config = Mock()
    monkeypatch.setattr(continuation.repair, 'test_config', test_config)
    monkeypatch.setattr(continuation, 'upload_folder', Mock())

    rows = continuation.test_existing(
        tmp_path, sha, 1376, Mock(), 'owner/repo')

    assert rows[0]['coco/bbox_mAP'] == 0.31
    test_config.assert_not_called()
