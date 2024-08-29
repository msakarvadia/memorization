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
from src.localize.weight.weight_utils import clm_loss_fn, count_num_params
from src.localize.weight.random_subnet import (
    get_base_edited_model,
    mask_model,
    GetSubnet,
    Conv1D,
    SupermaskConv,
)


def train(model, device, train_dataloader, optimizer, batch_size, loss_weighting):
    model.train()
    # train_dataloader = DataLoader(noise_data, batch_size=64, shuffle=False)
    # for batch_idx, (data, target) in enumerate(train_loader):
    for batch, label in train_dataloader:
        optimizer.zero_grad()
        model_output = model(batch, labels=batch)
        train_logits = model_output.logits
        # if we want to unlearn we just increase this loss (change direction in which we optimize)
        # train_loss = -model_output.loss
        loss = clm_loss_fn(batch, train_logits)
        loss_clean = loss[label == -1].mean()
        # print(loss_clean.shape)
        loss_noise = loss[label == 1].mean()
        # print(loss_noise.shape)
        # print(loss.shape)

        # NOTE(MS): this is the og loss funcation
        # loss *= -batch_size * label.to(device)
        # train_loss = loss.mean()
        loss = -(1 - loss_weighting) * loss_noise + loss_weighting * loss_clean
        train_loss = loss

        train_loss.backward()
        optimizer.step()


def do_random_greedy(
    model,
    noise_data,
    clean_data,
    n_layers,
    ratio,
    epochs=5,
    lr=0.1,
    momentum=0.9,
    weight_decay=0.0005,
    batch_size=64,
    loss_weighting=0.05,
    model_name="gpt2",
):
    clean_labels = [-1] * len(clean_data)
    noise_labels = [1] * len(noise_data)
    train_datasets = (noise_data, clean_data)
    train_labels = noise_labels + clean_labels

    train_data = torch.concat(train_datasets, dim=0)
    # want one train_datalaoder
    train_datas = []
    for i in range(len(train_labels)):
        train_datas.append([train_data[i], train_labels[i]])

    train_dataloader = DataLoader(train_datas, batch_size=batch_size, shuffle=True)

    # make model params grad frozen
    for name, param in model.named_parameters():
        param.requires_grad = False

    # model = mask_model(model, n_layers, ratio)
    model = mask_model(model, n_layers, ratio, model_name)

    optimizer = optim.SGD(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr,
        momentum=momentum,
        weight_decay=weight_decay,
    )

    for i in range(epochs):
        # print("EPOCH: ", i)
        train(model, device, train_dataloader, optimizer, batch_size, loss_weighting)

    model = get_base_edited_model(model, n_layers, model_name)
    return model
