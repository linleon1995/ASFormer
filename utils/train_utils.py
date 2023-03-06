import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

            
class ASFormerLoss():
    def __init__(self, num_class: int):
        self.num_class = num_class
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.mse = nn.MSELoss(reduction='none')

    def __call__(self, p, batch_target, mask):
        cls_loss = self.ce(p.transpose(2, 1).contiguous().view(-1, self.num_class),
                                        batch_target.view(-1))
        smooth_loss = torch.mean(torch.clamp(
            self.mse(F.log_softmax(p[:, :, 1:], dim=1), F.log_softmax(p.detach()[:, :, :-1], dim=1)), min=0,
            max=16) * mask[:, :, 1:])
        loss = cls_loss + 0.15*smooth_loss
        return loss
        