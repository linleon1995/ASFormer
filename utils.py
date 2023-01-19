import torch
import torch.nn as nn
import torch.nn.functional as F


def mse():
    mse = nn.MSELoss(reduction='none')


class ASFormerLoss:
    def __init__(self, num_classes):
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')
        self.num_classes = num_classes

    def loss(self, ps, batch_target, mask):
        loss = 0
        for p in ps:
            loss += self.ce(p.transpose(2, 1).contiguous().view(-1,
                            self.num_classes), batch_target.view(-1))
            loss += 0.15 * torch.mean(torch.clamp(
                self.mse(
                    F.log_softmax(p[:, :, 1:], dim=1),
                    F.log_softmax(p.detach()[:, :, :-1], dim=1)),
                min=0,
                max=16) * mask[:, :, 1:])
        return loss
