import inspect
import importlib
from typing import Dict

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import numpy as np


def find_class_in_module(module, cls_name):
    clsmembers = inspect.getmembers(module, inspect.isclass)
    for name, cls in clsmembers:
        if name == cls_name:
            return cls
    return None


def find_torch_lr_scheduler(name: str, strict: bool = True) -> callable:
    torch_lr_scheduler = importlib.import_module('torch.optim.lr_scheduler')
    lr_scheduler = find_class_in_module(torch_lr_scheduler, name)
    if lr_scheduler is None:
        if strict:
            raise ValueError(f'Unknown lr scheduler name: {name}.')
    return lr_scheduler


def find_torch_optimizer(name: str, strict: bool = True) -> callable:
    torch_optim = importlib.import_module('torch.optim')
    optimizer = find_class_in_module(torch_optim, name)
    if optimizer is None:
        if strict:
            raise ValueError(f'Unknown optimizer name: {name}.')
    return optimizer


def find_torch_loss(name: str, strict: bool = True) -> callable:
    torch_nn = importlib.import_module('torch.nn')
    loss = find_class_in_module(torch_nn, name)
    if loss is None:
        if strict:
            raise ValueError(f'Unknown loss name: {name}.')
    return loss


class BasePLModel(pl.LightningModule):
    def __init__(self, model: torch.nn.Module, optimizer_config: Dict,
                 lr_scheduler_config: Dict, loss_config: Dict):
        super().__init__()
        self.model = model  # Pytorch model
        self.optimizer_name = optimizer_config.pop('name')
        self.optimizer_config = optimizer_config
        self.optimizer_cls = find_torch_optimizer(self.optimizer_name)

        self.lr = lr_scheduler_config.pop('lr')
        self.lr_scheduler_name = lr_scheduler_config.pop('name')
        self.lr_scheduler_config = lr_scheduler_config
        self.lr_scheduler_cls = find_torch_lr_scheduler(self.lr_scheduler_name)

        self.loss_name = loss_config.pop('name')
        self.loss_config = loss_config
        self.loss_func_cls = find_torch_loss(self.loss_name, strict=False)
        if self.loss_func_cls is None:
            self.loss_func_cls = self.loss_config.pop('custom_loss')
        self.loss_func = self.loss_func_cls(**self.loss_config)

        self.save_hyperparameters()

    def configure_optimizers(self):
        optimizer = self.optimizer_cls(
            self.parameters(), lr=self.lr, **self.optimizer_config)
        lr_scheduler = self.lr_scheduler_cls(
            optimizer, **self.lr_scheduler_config)
        return {
            'optimizer': optimizer,
            'lr_scheduler': lr_scheduler
        }

    def training_step(self, train_batch):
        input, target = train_batch
        pred = self.model(input)
        loss = self.loss_func(pred, target)
        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch):
        input, target = val_batch
        pred = self.model(input)
        loss = self.loss_func(pred, target)
        self.log('val_loss', loss)

    def predict_step(self, batch):
        input, _ = batch
        pred = self.model(input)
        return pred


class PL_ASFormer(BasePLModel):
    def __init__(self,
                 model, optimizer_config, lr_scheduler_config, loss_config,
                 action_dict, sample_rate, results_dir):
        super().__init__(model, optimizer_config, lr_scheduler_config,
                         loss_config)

        self.action_to_id = action_dict
        self.id_to_action = {
            idx: action for action, idx in self.action_to_id.items()}

        self.sample_rate = sample_rate
        self.save_hyperparameters()

    def training_step(self, train_batch, batch_idx):
        # correct = 0
        # total = 0

        batch_input, batch_target, mask, vids = train_batch
        # batch_input, batch_target, mask = batch_input.to(
        #     device), batch_target.to(device), mask.to(device)
        ps = self.model(batch_input, mask)

        loss = 0
        for p in ps:
            loss += self.loss_func(p, batch_target, mask)

        # _, predicted = torch.max(ps.data[-1], 1)
        # correct += ((predicted == batch_target).float() *
        #             mask[:, 0, :].squeeze(1)).sum().item()
        # total += torch.sum(mask[:, 0, :]).item()

        # print("[epoch %d]: epoch loss = %f,   acc = %f" % (epoch + 1, epoch_loss / len(batch_gen.list_of_examples),
        #                                                    float(correct) / total))

        # if (epoch + 1) % 10 == 0 and batch_gen_tst is not None:
        #     self.test(batch_gen_tst, epoch)
        #     torch.save(self.model.state_dict(), save_dir +
        #                "/epoch-" + str(epoch + 1) + ".model")
        #     torch.save(self.optimizer.state_dict(), save_dir +
        #                "/epoch-" + str(epoch + 1) + ".opt")

        self.log('train_loss', loss)
        return loss

    def validation_step(self, val_batch, batch_idx):
        batch_input, batch_target, mask, _ = val_batch
        ps = self.model(batch_input, mask)
        loss = 0
        for p in ps:
            loss += self.loss_func(p, batch_target, mask)

        self.log('val_loss', loss)

#     def predict_step(self, batch, batch_idx, dataloader_idx=0):
#         self.model.to(device)
#         self.model.load_state_dict(torch.load(
#             model_dir + "/epoch-" + str(epoch) + ".model"))

#         batch_input, batch_target, mask, vids = batch
#         vid = vids[0]
# #                 print(vid)
#         features = np.load(features_path + vid.split('.')[0] + '.npy')
#         features = features[:, ::sample_rate]

#         input_x = torch.tensor(features, dtype=torch.float)
#         input_x.unsqueeze_(0)
#         input_x = input_x.to(device)
#         predictions = self.model(
#             input_x, torch.ones(input_x.size(), device=device))

#         for i in range(len(predictions)):
#             confidence, predicted = torch.max(
#                 F.softmax(predictions[i], dim=1).data, 1)
#             confidence, predicted = confidence.squeeze(), predicted.squeeze()

#             batch_target = batch_target.squeeze()
#             confidence, predicted = confidence.squeeze(), predicted.squeeze()

#             segment_bars_with_confidence(results_dir + '/{}_stage{}.png'.format(vid, i),
#                                             confidence.tolist(),
#                                             batch_target.tolist(), predicted.tolist())

#         recognition = []
#         for i in range(len(predicted)):
#             recognition = np.concatenate((recognition, [list(actions_dict.keys())[
#                 list(actions_dict.values()).index(
#                     predicted[i].item())]] * sample_rate))
#         f_name = vid.split('/')[-1].split('.')[0]
#         f_ptr = open(results_dir + "/" + f_name, "w")
#         f_ptr.write("### Frame level recognition: ###\n")
#         f_ptr.write(' '.join(recognition))
#         f_ptr.close()

#         f_name = vid[0].split('/')[-1].split('.')[0]
#         with open(f'{results_dir}/{f_name}.txt', "w+") as fw:
#             for pred_action in recognition:
#                 fw.write(f'{pred_action}\n')


# if __name__ == '__main__':
#     from model import MyTransformer
#     num_layers = 10
#     num_f_maps = 64
#     features_dim = 2048
#     # bz = 1
#     r1 = 2
#     r2 = 2
#     num_class = 6
#     channel_mask_rate = 0.3
#     torch_model = MyTransformer(
#         3, num_layers, r1, r2, num_f_maps, features_dim, num_class, channel_mask_rate
#     )

#     lr_scheduler_config = {
#         'name': 'ReduceLROnPlateau',
#         'lr': 1e-3,
#     }

#     optimizer_config = {
#         'name': 'Adadelta',
#         'eps': 1e-7
#     }

#     loss_config = {
#         'name': 'CrossEntropyLoss'
#     }

#     pl_model = BasePLModel(
#         torch_model,
#         optimizer_config,
#         lr_scheduler_config,
#         loss_config,
#         num_class=num_class
#     )
#     pass
