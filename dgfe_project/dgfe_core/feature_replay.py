"""Partial-forward design notes for DGFE/API.

The planned MMDetection partial replay is feature-level, not YOLO layer-cache
replay: reuse the clean backbone+neck feature tuple, perturb selected FPN
levels, then rerun only detector heads.
"""


def partial_forward_supported(detector=None) -> bool:
    """Report whether a detector exposes head-only loss replay."""
    return detector is not None and hasattr(detector, 'dgfe_replay_losses')
