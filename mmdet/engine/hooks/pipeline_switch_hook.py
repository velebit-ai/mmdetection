# Copyright (c) OpenMMLab. All rights reserved.
from mmcv.transforms import Compose
from mmengine.hooks import Hook
import torch

from mmdet.registry import HOOKS


@HOOKS.register_module()
class PipelineSwitchHook(Hook):
    """Switch data pipeline at switch_epoch.

    Args:
        switch_epoch (int): switch pipeline at this epoch.
        switch_pipeline (list[dict]): the pipeline to switch to.
    """

    def __init__(self, switch_epoch, switch_pipeline):
        self.switch_epoch = switch_epoch
        self.switch_pipeline = switch_pipeline
        self._restart_dataloader = False
        self._has_switched = False

    def before_train_epoch(self, runner):
        """switch pipeline."""
        epoch = runner.epoch
        train_loader = runner.train_dataloader
        if epoch >= self.switch_epoch and not self._has_switched:
            runner.logger.info('Switch pipeline now!')
            # The dataset pipeline cannot be updated when persistent_workers
            # is True, so we need to force the dataloader's multi-process
            # restart. This is a very hacky approach.
            train_loader.dataset.pipeline = Compose(self.switch_pipeline)
            if hasattr(train_loader, 'persistent_workers'
                       ) and train_loader.persistent_workers is True:
                train_loader._DataLoader__initialized = False
                train_loader._iterator = None
                self._restart_dataloader = True
            self._has_switched = True
        else:
            # Once the restart is complete, we need to restore
            # the initialization flag.
            if self._restart_dataloader:
                train_loader._DataLoader__initialized = True


@HOOKS.register_module()
class PipelineSwitchHookIter(Hook):
    """Switch data pipeline at switch_iter.

    Args:
        switch_iter (int): switch pipeline at this epoch.
        switch_pipeline (list[dict]): the pipeline to switch to.
    """

    def __init__(self, switch_iter, switch_pipeline):
        self.switch_iter = switch_iter
        self.switch_pipeline = switch_pipeline
        self._restart_dataloader = False
        self._has_switched = False

    def before_train_iter(self, runner, batch_idx, data_batch):
        """switch pipeline."""
        iteration = runner.iter
        train_loader = runner.train_dataloader
        if iteration >= self.switch_iter and not self._has_switched:
            runner.logger.info('Switch pipeline now!')
            # The dataset pipeline cannot be updated when persistent_workers
            # is True, so we need to force the dataloader's multi-process
            # restart. This is a very hacky approach.
            train_loader.dataset.pipeline = Compose(self.switch_pipeline)
            if hasattr(train_loader, 'persistent_workers'
                       ) and train_loader.persistent_workers is True:
                train_loader._DataLoader__initialized = False
                train_loader._iterator = None
                self._restart_dataloader = True
            self._has_switched = True
        else:
            # Once the restart is complete, we need to restore
            # the initialization flag.
            if self._restart_dataloader:
                train_loader._DataLoader__initialized = True


from mmengine.logging import MessageHub

@HOOKS.register_module()
class ValidationLossHook(Hook):
    def __init__(self):
        self.loss = {}
        self.cnt = 0

    def after_val_iter(self, runner, batch_idx, data_batch, outputs):
        model = runner.model
        data = model.data_preprocessor(data_batch, True)
        with torch.no_grad():
            losses = model(**data, mode='loss')
        parsed_losses, log_vars = model.parse_losses(losses)

        message_hub = MessageHub.get_current_instance()
        if batch_idx == 0:
            self.loss = {}
            for metric, value in log_vars.items():
                self.loss[metric] = value.item()
                #runner.logger.info(f"{metric}: {value}")
                
                message_hub.update_scalar(f'val/a', value.item())
                message_hub.update_scalar(f'train/a', value.item())
        else:
            for metric, value in log_vars.items():
                self.loss[metric] += value.item()
                #runner.logger.info(f"{metric}: {value}")
                message_hub.update_scalar(f'val/b', value.item())
                message_hub.update_scalar(f'train/b', value.item())

        self.cnt += 1

    def after_val(self, runner):
        runner.logger.info(self.loss)
        runner.logger.info(str(self.cnt))
        for metric, value in self.loss.items():
            self.loss[metric] = value / self.cnt

        runner.logger.info("###################\n"*3)
        runner.logger.info(self.loss)
