import inspect
import importlib
from typing import Dict

import pytorch_lightning as pl
import torch
import numpy as np


def find_class_in_module(module, cls_name):
    clsmembers = inspect.getmembers(module, inspect.isclass)
    for name, cls in clsmembers:
        if name == cls_name:
            return cls
    return None


def create_lr_scheduler(name: str) -> callable:
    torch_lr_scheduler = importlib.import_module('torch.optim.lr_scheduler')
    lr_scheduler = find_class_in_module(torch_lr_scheduler, name)
    if lr_scheduler is None:
        raise ValueError(f'Unknown lr scheduler name: {name}.')
    return lr_scheduler


def create_optimizer(name: str) -> callable:
    torch_optim = importlib.import_module('torch.optim')
    optimizer = find_class_in_module(torch_optim, name)
    if optimizer is None:
        raise ValueError(f'Unknown optimizer name: {name}.')
    return optimizer


def create_loss(name: str) -> callable:
    torch_nn = importlib.import_module('torch.nn')
    loss = find_class_in_module(torch_nn, name)
    if loss is None:
        raise ValueError(f'Unknown loss name: {name}.')
    return loss


class BasePLModel(pl.LightningModule):
    def __init__(self, torch_model: torch.nn.Module, optimizer_config: Dict,
                 lr_scheduler_config: Dict, loss_config: Dict, num_class: int):
        super().__init__()
        self.torch_model = torch_model  # Pytorch model
        self.optimizer_name = optimizer_config.pop('name')
        self.optimizer = create_optimizer(self.optimizer_name)

        self.lr = lr_scheduler_config.pop('lr')
        self.lr_scheduler_name = lr_scheduler_config.pop('name')
        self.lr_scheduler = create_lr_scheduler(self.lr_scheduler_name)
        self.lr_scheduler_config = lr_scheduler_config

        self.loss_name = loss_config.pop('name')
        self.loss_func = create_loss(self.loss_name)
        self.num_class = num_class

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = self.optimizer(self.parameters(), lr=self.lr)
        lr_scheduler = self.lr_scheduler(optimizer, **self.lr_scheduler_config)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler
        }

    def training_step(self, train_batch, batch_idx):
        input, target = train_batch
        pred = self.model(input)
        loss = self.loss_func(pred, target)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        input, target = val_batch
        pred = self.model(input)
        loss = self.loss_func(pred, target)
        self.log('val_loss', loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        input, target = batch
        pred = self.model(input)
        return pred


class PlModel(pl.LightningModule):
    def __init__(self,
                 model, optimizer_config, lr_scheduler_config, loss_config,
                 num_class, action_dict, sample_rate, results_dir):
        super().__init__(model, optimizer_config,
                         lr_scheduler_config, loss_config, num_class)

        self.action_to_id = action_dict
        self.id_to_action = {
            idx: action for action, idx in self.action_to_id.items()}

        self.sample_rate = sample_rate
        self.save_hyperparameters()

    def training_step(self, train_batch, batch_idx):
        input, target, mask, _ = train_batch
        pred = self.model(input, mask)
        loss = self.loss_func(pred, target, mask, mask)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        input, target, mask, _ = val_batch
        pred = self.model(input, mask)
        loss = self.loss_func(pred, target)
        self.log('val_loss', loss)

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        input, target, mask, vid = batch
        ms_pred = self.model(input, torch.ones(
            input.size(), device=self.device))
        _, pred = torch.max(ms_pred[-1].data, 1)

        for i in range(len(predictions)):
            confidence, predicted = torch.max(
                F.softmax(predictions[i], dim=1).data, 1)
            confidence, predicted = confidence.squeeze(), predicted.squeeze()

            batch_target = batch_target.squeeze()
            confidence, predicted = confidence.squeeze(), predicted.squeeze()

            segment_bars_with_confidence(results_dir + '/{}_stage{}.png'.format(vid, i),
                                         confidence.tolist(),
                                         batch_target.tolist(), predicted.tolist())

        recognition = []
        for i in range(len(predicted)):
            recognition = np.concatenate((recognition, [list(actions_dict.keys())[
                list(actions_dict.values()).index(
                    predicted[i].item())]] * sample_rate))

        pred = pred.squeeze()
        recognition = []
        for action_idx in pred:
            # action = [list(self.id_to_action.keys())[list(
            #     self.id_to_action.values()).index(action_idx.item())]]*self.sample_rate
            recognition.extend(
                [self.id_to_action[action_idx.item()]] * self.sample_rate)

        f_name = vid[0].split('/')[-1].split('.')[0]
        with open(f'{self.results_dir}/{f_name}.txt', "w+") as fw:
            for pred_action in recognition:
                fw.write(f'{pred_action}\n')

        return pred


if __name__ == '__main__':
    from model import MyTransformer
    num_layers = 10
    num_f_maps = 64
    features_dim = 2048
    # bz = 1
    r1 = 2
    r2 = 2
    num_class = 6
    channel_mask_rate = 0.3
    torch_model = MyTransformer(
        3, num_layers, r1, r2, num_f_maps, features_dim, num_class, channel_mask_rate
    )

    lr_scheduler_config = {
        'name': 'ReduceLROnPlateau',
        'lr': 1e-3,
    }

    optimizer_config = {
        'name': 'Adadelta',
    }

    loss_config = {
        'name': 'CrossEntropyLoss'
    }

    pl_model = BasePLModel(
        torch_model,
        optimizer_config,
        lr_scheduler_config,
        loss_config,
        num_class=num_class
    )
