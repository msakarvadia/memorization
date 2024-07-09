import sys
import argparse
from src.localize.neuron.neuron_utils import (
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

"""
from zero_out import fast_zero_out_vector
from slimming import patch_slim, reinit_slim, compute_l1_loss, slim
from hard_concrete import (
    L0Mask,
    MaskedLinear,
    patch_hardconcrete,
    reinit_hardconcrete,
    transpose_conv1d,
    compute_total_regularizer,
    get_sparsity,
    hard_concrete,
)
"""
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

"""# Activations"""


def register_hook(model, layer_idx, ori_activations, attr_str):
    ff_layer = get_attributes(model, attr_str)

    # print(attr_str)
    # print("input shape in hook: ", ori_activations.shape)

    def hook_fn(m, i, o):
        # print(ori_activations.shape)
        # print("o dim: ", o.shape)
        ori_activations[layer_idx] = o.squeeze().cpu()
        # ori_activations[layer_idx] = o.cpu()

    return ff_layer.register_forward_hook(hook_fn)


@torch.no_grad()
def get_ori_activations_ACT(inner_dim, model, inputs):
    # seq_len = inputs['input_ids'].shape[1]
    seq_len = inputs.shape[1]
    batch_size = inputs.shape[0]
    ori_activations = torch.zeros(
        (model.config.n_layer, batch_size, seq_len, inner_dim)
    )

    handles = []
    for ly in range(model.config.n_layer):
        handle = register_hook(
            model,
            ly,
            ori_activations,
            f"{model.attr_dict['transformer_layer']}.{ly}.{model.attr_dict['ffn_act']}",
        )
        handles.append(handle)

    # print("input shape: ", inputs.shape)
    out = model(inputs)
    # print("here")
    for handle in handles:  # detach the hooks
        handle.remove()

    return ori_activations


def largest_act(inner_dim, model, inputs, gold_set, model_name="gpt2", prompt_len=50):

    @torch.no_grad()
    def get_ffn_norms():
        all_norms = torch.zeros((model.config.n_layer, inner_dim))
        for ly in range(model.config.n_layer):
            attr_str = f"{model.attr_dict['transformer_layer']}.{ly}.{model.attr_dict['ffn_out']}.weight"
            weights = get_attributes(model, attr_str)
            if "gpt2" in model_name:
                norms = torch.norm(weights, dim=1)
            else:
                norms = torch.norm(weights, dim=0)
            all_norms[ly] = norms.cpu()

        return all_norms

    all_norms = get_ffn_norms()
    # print("norms shape: ", all_norms.shape)

    # prompt_start_i = args.prompt_len -1 if hasattr(args, 'prompt_len') else 0  # -1 for 0-indexed
    prompt_start_i = prompt_len - 1

    activations = get_ori_activations_ACT(inner_dim, model, inputs)
    # print(activations.shape)
    activations = activations[
        :, :, prompt_start_i:-1
    ]  # [n_layer, suffix_len, inner_dim]
    all_norms = get_ffn_norms()

    # print(activations.shape)
    # print(activations.mean(1).shape)

    act_mean = (
        activations.mean(1).mean(1).cpu().abs() * all_norms
    )  # Average once over batch size, then again over suffix length
    # torch.save(act_mean, os.path.join(args.out_dir, 'act-mean.pt'))
    # if gold_set is not None:
    #    score = get_layerwise_scores(act_mean, gold_set, args.ratio)
    return act_mean
