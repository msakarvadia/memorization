# General structure from https://github.com/pytorch/examples/blob/master/mnist/main.py
from __future__ import print_function
import argparse
import os
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import CosineAnnealingLR
import torch.autograd as autograd
import copy

device = "cuda" if torch.cuda.is_available() else "cpu"
from torch.utils.data import DataLoader
from src.localize.neuron.neuron_utils import (
    track_all_metrics,
)

args = None

class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int(k * scores.numel())

        # flat_out and out access the same memory.
        flat_out = out.flatten()
        flat_out[idx[:j]] = 0
        flat_out[idx[j:]] = 1

        return out

    @staticmethod
    def backward(ctx, g):
        # send the gradient g straight-through on the backward pass.
        return g, None

class Conv1D(nn.Module):
    """
    1D-convolutional layer as defined by Radford et al. for OpenAI GPT (and also used in GPT-2).

    Basically works like a linear layer but the weights are transposed.

    Args:
        nf (`int`): The number of output features.
        nx (`int`): The number of input features.
    """

    def __init__(self, nf, nx):
        super().__init__()
        self.nf = nf
        self.weight = nn.Parameter(torch.empty(nx, nf))
        self.bias = nn.Parameter(torch.zeros(nf))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), self.weight)
        x = x.view(size_out)
        return x

class SupermaskConv(Conv1D):
    def __init__(self, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # initialize the scores
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        #NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False
        self.bias.requires_grad = False

    def forward(self, x):
        sparsity = 0.001
        subnet = GetSubnet.apply(self.scores.abs(), sparsity)
        w = self.weight * subnet

        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), w)
        x = x.view(size_out)
        return x


def mask_model(model, n_layers):
    for layer in range(n_layers):
      # make mask
      mask = SupermaskConv(512, 128).to(device)
      # assign old weights to mask
      mask.weight = model.transformer.h[layer].mlp.c_fc.weight
      mask.bias = model.transformer.h[layer].mlp.c_fc.bias
      # assign mask to layer
      model.transformer.h[layer].mlp.c_fc = copy.deepcopy(mask)

      # make mask
      mask = SupermaskConv(128, 512).to(device)
      # assign old weights to mask
      mask.weight = model.transformer.h[layer].mlp.c_proj.weight
      mask.bias = model.transformer.h[layer].mlp.c_proj.bias
      # assign mask to layer
      model.transformer.h[layer].mlp.c_proj = copy.deepcopy(mask)

    return model


def train(model, device, noise_data, optimizer):
    model.train()
    train_dataloader = DataLoader(noise_data, batch_size=64, shuffle=False)
    #for batch_idx, (data, target) in enumerate(train_loader):
    for batch in train_dataloader:
        optimizer.zero_grad()
        model_output = model(batch, labels=batch)
        train_logits = model_output.logits
        train_loss = model_output.loss

        #data, target = data.to(device), target.to(device)

        #utput = model(data)
        #loss = criterion(output, target)
        train_loss.backward()
        optimizer.step()

def do_random(model, noise_data, n_layers, ratio):
    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=0.1,
        momentum=0.9,
        weight_decay=0.0005,
    )

    model = mask_model(model, n_layers)

    epochs = 100
    for i in range(epochs):
        print("EPOCH: ", i)
        train(model, device, noise_data, optimizer)

    return model
