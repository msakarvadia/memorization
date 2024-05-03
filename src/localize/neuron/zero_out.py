import argparse
import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import numpy as np

# from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel

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

from tqdm import tqdm
import copy
import math

device = "cuda" if torch.cuda.is_available() else "cpu"

# from random import randrange, choices, sample
# from operator import add

# from collections import OrderedDict
# from typing import Dict, Callable
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


"""# Implement Localization Strategy: Zero-out

We will implement a few of the strategies from this paper:
 https://arxiv.org/pdf/2311.09060.pdf

 Will need to track memorization before and after localization, and accuracy and perplexity before and after removing memorization on target task.
"""


@torch.no_grad()
def fast_zero_out_vector(
    inner_dim, n_batches, model, inputs, labels, prompt_len, gold_set=None
):
    model.eval()
    loss_ori = model(inputs, labels=labels).loss.item()

    losses = torch.zeros((model.config.n_layer, inner_dim))
    seq_len = inputs.shape[1]
    print("seq len: ", seq_len)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

    for ly in tqdm(range(model.config.n_layer)):
        inner_losses = []

        inner_dim = model.inner_dim
        attr_str = (
            f"{model.attr_dict['transformer_layer']}.{ly}.{model.attr_dict['ffn_act']}"
        )

        for zero_idx in range(inner_dim):
            mask = torch.zeros(inner_dim)
            mask[zero_idx] = 1

            patch_ff_layer(
                model,
                attr_str,
                onehot_coef=mask.to(device),
            )

            batch_loss = model(
                inputs, labels=labels
            ).loss.item()  # [bs, seq_len, vocab]

            inner_losses.append(batch_loss)

            unpatch_ff_layer(
                model,
                attr_str,
            )

        # print(inner_losses)
        print("layer: ", ly)
        losses[ly] = torch.tensor(inner_losses)

    # print(losses)
    delta_losses = losses - loss_ori

    # if gold_set is not None:
    #    score = get_layerwise_scores(delta_losses, gold_set, ratio)
    return delta_losses
