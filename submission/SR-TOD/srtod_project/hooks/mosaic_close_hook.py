from mmdet.registry import HOOKS
from mmengine.hooks import Hook


@HOOKS.register_module()
class MosaicCloseHook(Hook):
    """Disable Mosaic for the final epochs without changing model loss."""

    def __init__(self, num_last_epochs=10, skip_type_keys=('Mosaic',)):
        self.num_last_epochs = num_last_epochs
        self.skip_type_keys = skip_type_keys
        self._switched = False

    def before_train_epoch(self, runner):
        if self._switched:
            return
        if (runner.epoch + 1) < runner.max_epochs - self.num_last_epochs:
            return
        runner.train_dataloader.dataset.update_skip_type_keys(self.skip_type_keys)
        self._switched = True
        runner.logger.info('Mosaic disabled for the final %d epochs.', self.num_last_epochs)
