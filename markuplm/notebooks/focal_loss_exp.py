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
class WeightedFocalLoss(nn.Module):
    "Non weighted version of Focal Loss"
    def __init__(self, alpha=.25, gamma=2):
        super(WeightedFocalLoss, self).__init__()
        self.alpha = torch.tensor([alpha, 1-alpha])
        self.gamma = gamma

    def format_inputs(self, inputs):
        inputs = torch.softmax(inputs, dim=2)
        inputs = inputs.T[0]
        return inputs

    def forward(self, inputs, targets):

        # inputs = self.format_inputs(inputs)

        # targets = targets.type(torch.float)

        # targets = targets.clip(0)  #? Change -100 to 0

        # BCE_loss = F.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        

        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        # print("BCE_loss: ", BCE_loss)

        targets = targets.type(torch.long)
        self.alpha = self.alpha.to(device=inputs.device)
        
        at = self.alpha.gather(0, targets.clip(0).data.view(-1))

        pt = torch.exp(-BCE_loss)
        F_loss = at*(1-pt)**self.gamma * BCE_loss
        # print("F_loss: ",F_loss)
        # print("F_loss_mean: ", F_loss.mean()) #!Maybe add here a Raise Error in case the loss is inf
        return F_loss.mean()

# %%
# labels = torch.FloatTensor([1, 0, 1, 0]).to(torch.long)
# preds = torch.FloatTensor([[0.9, 0.1], [0.9, 0.1], [0.1, 0.1], [0.1, 0.1]]).to(torch.float)

labels = torch.FloatTensor([1]).to(torch.long)
preds = torch.FloatTensor([[0.9, 0.1]]).to(torch.float)

print(preds, preds.shape, preds.type())
print(labels, labels.shape, labels.type())


# %%
f_loss = WeightedFocalLoss()
ce_loss = CrossEntropyLoss()

# %%
print(preds)
print(labels)

ce_loss_value = ce_loss(preds, labels)
print(f"ce_loss_value:\n {ce_loss_value}")

f_loss_value = f_loss(preds, labels)
print(f"f_loss_value:\n {f_loss_value}")

print(f"ratio:\n {ce_loss_value/f_loss_value}")

# %%
import numpy as np
alpha = 0.25
gamma = 2
x = 0.1
pos_f_loss =  -alpha * (1 - x)**gamma * np.log(x) #For positives
# neg_f_loss = -(1-alpha) * x**gamma * np.log(1-x) #For negatives
ce_loss = -np.log(1-x)
print(pos_f_loss)
# print(neg_f_loss)
print(ce_loss)


# %%
print(preds)
print(labels)

ce_loss_value = ce_loss(preds, labels)
print(f"ce_loss_value:\n {ce_loss_value}")

f_loss_value = f_loss(preds, labels)
print(f"f_loss_value:\n {f_loss_value}")

print(f"ratio:\n {ce_loss_value/f_loss_value}")

# %%
print(preds)
print(labels)

ce_loss_value = ce_loss(preds, labels)
print(f"ce_loss_value:\n {ce_loss_value}")

f_loss_value = f_loss(preds, labels)
print(f"f_loss_value:\n {f_loss_value}")

print(f"ratio:\n {ce_loss_value/f_loss_value}")

# %%
print(preds)
print(labels)

ce_loss_value = ce_loss(preds, labels)
print(f"ce_loss_value:\n {ce_loss_value}")

f_loss_value = f_loss(preds, labels)
print(f"f_loss_value:\n {f_loss_value}")

print(f"ratio:\n {ce_loss_value/f_loss_value}")

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
