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
from tqdm import tqdm

device = "cuda" if torch.cuda.is_available() else "cpu"
from torch.utils.data import DataLoader
from src.localize.neuron.neuron_utils import (
    track_all_metrics,
)


class GetSubnet(autograd.Function):
    @staticmethod
    def forward(ctx, scores, k):
        # Get the supermask by sorting the scores and using the top k%
        out = scores.clone()
        _, idx = scores.flatten().sort()
        j = int((k) * scores.numel())

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
    def __init__(self, sparsity, model_name, *args, **kwargs):

        super().__init__(*args, **kwargs)

        # initialize the scores
        # self.scores = nn.Parameter(torch.Tensor(self.weight.size()).to(self.weight.device))
        self.scores = nn.Parameter(torch.Tensor(self.weight.size()))
        nn.init.kaiming_uniform_(self.scores, a=math.sqrt(5))

        # NOTE: initialize the weights like this.
        nn.init.kaiming_normal_(self.weight, mode="fan_in", nonlinearity="relu")

        # NOTE: turn the gradient on the weights off
        self.weight.requires_grad = False
        self.bias.requires_grad = False

        # set sparsity
        self.sparsity = sparsity

        # set model name
        self.model_name = model_name

    def forward(self, x):
        # NOTE(MS): (need to move subnet to same device as weight)
        subnet = GetSubnet.apply(self.scores.abs(), self.sparsity).to(
            self.weight.device
        )
        if "gpt" in self.model_name:
            w = self.weight * subnet
        if "pythia" in self.model_name:
            w = self.weight.T * subnet
        # NOTE(ms): need to ensure dtype match
        w = w.to(self.weight.dtype)

        size_out = x.size()[:-1] + (self.nf,)
        x = torch.addmm(self.bias, x.view(-1, x.size(-1)), w)
        x = x.view(size_out)
        return x


def mask_model(model, n_layers, ratio, model_name="gpt2"):
    for layer in range(n_layers):
        if "gpt" in model_name:
            # make mask
            mask = SupermaskConv(ratio, model_name, 512, 128).to(device)
            # assign old weights to mask
            mask.weight = model.transformer.h[layer].mlp.c_fc.weight
            mask.bias = model.transformer.h[layer].mlp.c_fc.bias
            # assign mask to layer
            model.transformer.h[layer].mlp.c_fc = copy.deepcopy(mask)

            # make mask
            mask = SupermaskConv(ratio, model_name, 128, 512).to(device)
            # assign old weights to mask
            mask.weight = model.transformer.h[layer].mlp.c_proj.weight
            mask.bias = model.transformer.h[layer].mlp.c_proj.bias
            # assign mask to layer
            model.transformer.h[layer].mlp.c_proj = copy.deepcopy(mask)
        if "pythia" in model_name:
            # make mask
            # TODO (MS): don't hardcode pythia dims cus many different model sizes
            mask = SupermaskConv(ratio, model_name, 16384, 4096).to(device)
            # assign old weights to mask
            mask.weight = model.gpt_neox.layers[layer].mlp.dense_h_to_4h.weight
            mask.bias = model.gpt_neox.layers[layer].mlp.dense_h_to_4h.bias
            # assign mask to layer
            model.gpt_neox.layers[layer].mlp.dense_h_to_4h = copy.deepcopy(mask)

            # make mask
            mask = SupermaskConv(ratio, model_name, 4096, 16384).to(device)
            # assign old weights to mask
            mask.weight = model.gpt_neox.layers[layer].mlp.dense_4h_to_h.weight
            mask.bias = model.gpt_neox.layers[layer].mlp.dense_4h_to_h.bias
            # assign mask to layer
            model.gpt_neox.layers[layer].mlp.dense_4h_to_h = copy.deepcopy(mask)
        print("Masked layer: ", layer)

    return model


def get_base_edited_model(model, n_layers, model_name):
    """This is how we merge the mask into the base weights
    rather than, having a scores attribute"""

    for layer in range(n_layers):
        if "gpt" in model_name:
            # grab mask
            mask = model.transformer.h[layer].mlp.c_fc
            # assign edited weights to base model
            subnet = GetSubnet.apply(mask.scores.abs(), mask.sparsity)
            w = mask.weight * subnet
            # original layer
            model.transformer.h[layer].mlp.c_fc = Conv1D(512, 128).to(device)
            # assign edited weight to base model
            model.transformer.h[layer].mlp.c_fc.weight = torch.nn.Parameter(w)
            # assign bias to base model
            model.transformer.h[layer].mlp.c_fc.bias = mask.bias

            # grab mask
            mask = model.transformer.h[layer].mlp.c_proj
            # assign edited weights to base model
            subnet = GetSubnet.apply(mask.scores.abs(), mask.sparsity)
            w = mask.weight * subnet
            # original layer
            model.transformer.h[layer].mlp.c_proj = Conv1D(128, 512).to(device)
            # assign edited weight to base model
            model.transformer.h[layer].mlp.c_proj.weight = torch.nn.Parameter(w)
            # assign bias to base model
            model.transformer.h[layer].mlp.c_proj.bias = mask.bias
        if "pythia" in model_name:
            mask = model.gpt_neox.layers[layer].mlp.dense_h_to_4h
            # assign edited weights to base model
            subnet = GetSubnet.apply(mask.scores.abs(), mask.sparsity).to(
                mask.weight.device
            )
            w = mask.weight.T * subnet
            # original layer
            model.gpt_neox.layers[layer].mlp.dense_h_to_4h = Conv1D(4096, 16384).to(
                device
            )
            # assign edited weight to base model
            model.gpt_neox.layers[layer].mlp.dense_h_to_4h.weight = torch.nn.Parameter(
                w
            )
            # assign bias to base model
            model.gpt_neox.layers[layer].mlp.dense_h_to_4h.bias = mask.bias

            # grab mask
            mask = model.gpt_neox.layers[layer].mlp.dense_4h_to_h
            # assign edited weights to base model
            subnet = GetSubnet.apply(mask.scores.abs(), mask.sparsity).to(
                mask.weight.device
            )
            w = mask.weight.T * subnet
            # original layer
            model.gpt_neox.layers[layer].mlp.dense_4h_to_h = Conv1D(16384, 4096).to(
                device
            )
            # assign edited weight to base model
            model.gpt_neox.layers[layer].mlp.dense_4h_to_h.weight = torch.nn.Parameter(
                w
            )
            # assign bias to base model
            model.gpt_neox.layers[layer].mlp.dense_4h_to_h.bias = mask.bias

    return model


def train(model, device, noise_data, optimizer, batch_size):
    model.train()
    train_dataloader = DataLoader(noise_data, batch_size=batch_size, shuffle=False)
    # for batch_idx, (data, target) in enumerate(train_loader):
    for batch in tqdm(train_dataloader):
        optimizer.zero_grad()
        model_output = model(batch, labels=batch)
        train_logits = model_output.logits
        # if we want to unlearn we just increase this loss (change direction in which we optimize)
        train_loss = -model_output.loss

        train_loss.backward()
        optimizer.step()


def do_random(
    model,
    noise_data,
    n_layers,
    ratio,
    epochs=5,
    lr=0.1,
    momentum=0.9,
    weight_decay=0.0005,
    model_name="gpt2",
    batch_size=64,
):

    # make model params grad frozen
    for name, param in model.named_parameters():
        param.requires_grad = False

    model = mask_model(model, n_layers, ratio, model_name)

    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )
    if "pythia" in model_name:
        # less memory intensive
        optimizer = optim.SGD(
            [p for p in model.parameters() if p.requires_grad],
            lr=lr,
            # momentum=momentum,
            # weight_decay=weight_decay,
        )

    for i in range(epochs):
        print("EPOCH: ", i)
        train(
            model,
            device,
            noise_data,
            optimizer,
            batch_size,
        )

    model = get_base_edited_model(model, n_layers, model_name)
    return model
