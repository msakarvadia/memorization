import sys
import argparse
from neuron_utils import (
    get_attr_str,
    set_model_attributes,
    get_attributes,
    Patch,
    set_attributes,
    patch_ff_layer,
    unpatch_ff_layer,
    refined_check_percent_memorized,
    accuracy,
    perplexity,
    loss,
    compute_average_metric_accross_dataset,
    track_all_metrics,
    get_model,
    apply_ablation_mask_to_neurons,
    remove_ablation_mask_from_neurons,
    apply_mean_ablation_mask_to_neurons,
    apply_noise_ablation_mask_to_neurons,
)

from zero_out import fast_zero_out_vector
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import numpy as np
from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel

from tqdm import tqdm
import copy
import math


device = "cuda" if torch.cuda.is_available() else "cpu"

from random import randrange, choices, sample
from operator import add

from collections import OrderedDict
from typing import Dict, Callable
import torch
from transformers.pytorch_utils import Conv1D
import torch.nn.init as init

import random


torch.__version__
torch.manual_seed(0)
random.seed(0)

# Constants
num_test = 1000
max_ctx = 150
batch_size = 1000
DATA_SEED = 598


"""# Slimming"""


def patch_slim(model):
    for ly in range(model.config.n_layer):
        ff_attrs = (
            f"{model.attr_dict['transformer_layer']}.{ly}.{model.attr_dict['ffn_act']}"
        )
        patch_ff_layer(
            model,
            ff_attrs,
            intermediate_size=model.inner_dim,
        )


def reinit_slim(model):
    for ly in range(model.config.n_layer):
        attrs_str = f"{model.attr_dict['transformer_layer']}.{ly}.{model.attr_dict['ffn_act']}.slim_coef"
        coef = get_attributes(model, attrs_str)
        init.ones_(coef)


def compute_l1_loss(model, start_layer_idx):
    l1_loss = 0.0
    N = 0
    cnt = 0
    for m in model.modules():
        if isinstance(m, Patch):
            cnt += 1
            if cnt > start_layer_idx:
                l1_loss = l1_loss + torch.norm(m.slim_coef, 1)
                N += len(m.slim_coef)
    l1_loss /= N
    return l1_loss


def slim(lr, epoch, lambda_l1, stop_loss, threshold, model, inputs, gold_set):
    model.eval()

    # start_layer_idx = args.start_mask_layer if hasattr(args, 'start_mask_layer') else 0
    start_layer_idx = 0
    # set tunable parameters
    cnt = 0
    params = []
    for n, p in model.named_parameters():
        if "slim" in n:
            cnt += 1
            if cnt > start_layer_idx:
                p.requires_grad = True
                print(n)
            else:
                p.requires_grad = False
            params.append(p)
        else:
            p.requires_grad = False
    print("-" * 100)

    # print(params)
    optimizer = torch.optim.Adam(params, lr=lr)

    # training
    scores, reg_losses, lm_losses = [], [], []
    for i in tqdm(range(epoch)):
        optimizer.zero_grad()

        outputs = model(inputs, labels=inputs)
        l1_loss = compute_l1_loss(model, start_layer_idx)
        # print("loss: ", l1_loss)

        lm_loss = outputs.loss
        # print("lm loss: ", lm_loss)
        loss = lm_loss + lambda_l1 * l1_loss

        if (i + 1) % 10 == 0:
            ckpt_params = torch.stack(params).clamp(min=0.0, max=1.0)
            sparsity = (ckpt_params[start_layer_idx:] < threshold).float().mean().item()
            print(
                i + 1, f"lm loss: {lm_loss.item():.3f}, l1 loss: {l1_loss.item():.2f}"
            )
            print("  Sparsity:", sparsity)
            # if gold_set:
            #    score = get_layerwise_scores(ckpt_params, gold_set, args.ratio)
            # else:
            #    score = 0 # dummy
            #    if args.save_ckpt: save_params(args, ckpt_params, f'{i+1}.pt')
            # scores.append(score)
            lm_losses.append(lm_loss.item())
            reg_losses.append(l1_loss.item())
            if l1_loss < stop_loss:
                break

        loss.backward()
        optimizer.step()

    params = torch.stack(params).clamp(min=0.0, max=1.0).detach().cpu()

    return params
