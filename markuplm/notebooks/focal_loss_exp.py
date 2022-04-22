# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.13.6
#   kernelspec:
#     display_name: Python 3.7.11 ('markuplmft')
#     language: python
#     name: python3
# ---

# %%
import torch
from torch import nn
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
import torch.nn as nn


# %%
class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"

    def __init__(self, alpha: float = 0.25, gamma: float = 2, label_smoothing: float = 0.0):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1 - alpha])
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def format_inputs(self, inputs):
        inputs = inputs.T[0]
        return inputs

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(
            inputs, targets, reduction="none", label_smoothing=self.label_smoothing
        )

        targets = targets.type(torch.long)
        self.alpha = self.alpha.to(device=inputs.device)

        at = self.alpha.gather(0, targets.clip(0).data.view(-1))

        pt = torch.exp(-BCE_loss)
        F_loss = at * (1 - pt) ** self.gamma * BCE_loss
        print(f"F_loss: {F_loss}")
        F_loss = F_loss[F_loss.nonzero()]
        print(f"BCE_loss: \n {BCE_loss}")
        print(f"inputs: \n {inputs}")
        print(f"targets: \n {targets}")
        print(f"at: \n {at}")
        print(f"pt: \n {pt}")
        print(f"F_loss: \n {F_loss}")
        return F_loss.mean()


# %%
targets = torch.FloatTensor([[1, 0, -100,1, 0, -100]])
inputs = torch.FloatTensor([[[0.9, 0.1], [0.1, 0.9], [0.5, 0.5], [0.01, 0.99], [0.99, 0.01], [0.5, 0.5]]])

inputs = inputs.to(torch.float)
targets = targets.to(torch.long)

def ptensor(tensor):
    print(f" type: {tensor.type()} \n shape: {tensor.shape} \n tensor:\n{tensor}")
ptensor(inputs)
print()
ptensor(targets)
print('-'*10)

inputs = inputs.view(-1, 2)
targets = targets.view(-1)
ptensor(inputs)
print()
ptensor(targets)


# %%
torch.softmax(inputs, dim=1)

# %%
f_loss = WeightedFocalLoss()
ce_loss = CrossEntropyLoss()

# %%
# ce_loss_value = ce_loss(inputs, targets)
# print(f"ce_loss_value:\n {ce_loss_value}")

f_loss_value = f_loss(inputs, targets)
print(f"f_loss_value:\n {f_loss_value}")

# print(f"ratio:\n {ce_loss_value/f_loss_value}")

# %%
