"""Boxgrad adapter helpers.

Model-specific heads may override localization-loss selection; the generic
detector still owns perturbation construction.
"""


def localization_loss_names(losses: dict, prefixes: tuple[str, ...] = ()) -> set[str]:
    names = set()
    for name, value in losses.items():
        if 'loss' not in name:
            continue
        if prefixes and not name.startswith(prefixes):
            continue
        if isinstance(value, (list, tuple)) or hasattr(value, 'mean'):
            name_l = name.lower()
            if any(key in name_l for key in ('bbox', 'box', 'iou', 'reg', 'l1')):
                names.add(name)
    return names

