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
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha])
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')

        targets = targets.type(torch.long)
        at = self.alpha.gather(0, targets.data.view(-1))
        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        print(f"BCE_loss:\n{BCE_loss}")
        print(f"F_loss:\n{F_loss}")
        return F_loss.mean()


# %%
# preds = torch.FloatTensor([0.9, 0.1, 0.1, 0.9])
# labels = torch.FloatTensor([1, 1, 0, 0])

preds = torch.FloatTensor([[0.9], [0.9], [0.1], [0.1]])
labels = torch.FloatTensor([[1], [0], [1], [0]])

# preds = torch.FloatTensor([[0.9, 0.1], [0.9, 0.1], [0.1, 0.1], [0.1, 0.1]])
# labels = torch.FloatTensor([[1, 0], [0, 1], [1, 0], [0, 1]])

# preds = torch.FloatTensor([1])
# labels = torch.FloatTensor([1])

# preds = preds.reshape(1,-1)
# labels = labels.reshape(1,-1)

print(preds, preds.shape)
print(labels, labels.shape)

# %%
f_loss = WeightedFocalLoss()
ce_loss = CrossEntropyLoss()

# %%
import pandas as pd

pd.DataFrame([preds.tolist(), labels.tolist()], ["pred", "labels"])

# %%
ce_loss_value = ce_loss(preds, labels)
print(f"ce_loss_value:\n {ce_loss_value}")
f_loss_value = f_loss(preds, labels)
print(f"f_loss_value:\n {f_loss_value}")
print(f"ratio:\n {ce_loss_value/f_loss_value}")

# %%
ce_loss_value = ce_loss(preds, labels)
print(f"ce_loss_value:\n {ce_loss_value}")
f_loss_value = f_loss(preds, labels)
print(f"f_loss_value:\n {f_loss_value}")
print(f"ratio:\n {ce_loss_value/f_loss_value}")

# %%

# %%
import numpy as np
x = 0.9

-np.log(x)
-np.log(1-x)

alpha = 0.25
gamma = 2

pos = -alpha * np.power((1 - x), gamma) * np.log(x)
neg = -(1-alpha) * np.power(x, gamma) * np.log(1-x)
print(pos, neg)
print(-np.log(x))

# %%
criteria = nn.CrossEntropyLoss(label_smoothing=0.1)
input = torch.tensor([[0.90, 0.10], [0.90, 0.10]],dtype=torch.float)
target = torch.tensor([[0, 0], [1, 0]], dtype=torch.float)
criteria(input, target)

# %%
