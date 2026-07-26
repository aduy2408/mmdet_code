from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmdet.registry import HOOKS


@HOOKS.register_module()
class LTMRWeightWarmupHook(Hook):
    """Linearly warm LTMR loss weight using the global iteration."""

    priority = 'NORMAL'

    def __init__(self, target_weight: float = 0.05,
                 warmup_ratio: float = 0.1) -> None:
        self.target_weight = target_weight
        self.warmup_ratio = warmup_ratio

    def before_train_iter(self, runner, batch_idx: int,
                          data_batch=None) -> None:
        model = runner.model.module if is_model_wrapper(
            runner.model) else runner.model
        warmup_iters = max(1, round(runner.max_iters * self.warmup_ratio))
        weight = self.target_weight * min(1.0, runner.iter / warmup_iters)
        model.bbox_head.ltmr_weight = weight
        runner.message_hub.update_scalar('train/ltmr_weight', weight)
