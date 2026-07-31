import random

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
