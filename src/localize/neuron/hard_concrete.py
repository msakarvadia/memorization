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


class L0Mask(torch.nn.Module):
    def __init__(self, mask_dim, mask_p, beta, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.mask_setting = "mask"
        self.mask_scores = torch.nn.Parameter(torch.zeros(mask_dim))
        self.mask_p = mask_p
        self.b = beta  # temerature (0,1); b->0, Bernoulli
        self.l, self.r = -0.1, 1.1
        self.is_train = True
        self.init_weights()

    def init_weights(self):
        p = (self.mask_p - self.l) / (self.r - self.l)
        init.constant_(self.mask_scores, val=np.log(p / (1 - p)))
        init.normal_(self.mask_scores, mean=0, std=0.01)

    def produce_mask(self, is_train_runtime=True):
        if self.is_train and is_train_runtime:
            u = torch.zeros_like(self.mask_scores).uniform_().clamp(0.0001, 0.9999)
            s = torch.sigmoid((u.log() - (1 - u).log() + self.mask_scores) / self.b)
        else:
            s = torch.sigmoid(self.mask_scores)
        s_bar = s * (self.r - self.l) + self.l  # (-0.1, 1.1)
        mask = s_bar.clamp(min=0.0, max=1.0)
        return mask

    def regularizer(self):
        return (
            torch.sum(
                torch.sigmoid(self.mask_scores - self.b * np.log(-self.l / self.r))
            )
            / self.mask_scores.numel()
        )


class MaskedLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        out_w_per_mask: int,
        in_w_per_mask: int,
        mask_p: float,
        beta: float,
        layer_idx: int,
        bias: bool = True,
    ):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.out_w_per_mask = out_w_per_mask
        self.in_w_per_mask = in_w_per_mask

        assert out_features % out_w_per_mask == 0, "{} % {} not 0".format(
            out_features, out_w_per_mask
        )
        assert in_features % in_w_per_mask == 0, "{} % {} not 0".format(
            in_features, in_w_per_mask
        )
        mask_dim = (1, out_features // out_w_per_mask, 1, in_features // in_w_per_mask)
        self.mask = L0Mask(mask_dim, mask_p, beta, layer_idx)

        self.cached_activation = None
        self.do_caching = False

    def produce_mask(self):
        mask = self.mask.produce_mask()
        return mask

    def forward(self, input: torch.tensor):
        # input: [bs, seqlen, 3072], weight: [768, 3072]
        # [1, 1, 1, 3072] * [768, 1, 1, 3072]
        masked_weight = self.produce_mask() * self.weight.reshape(
            self.out_w_per_mask,
            self.out_features // self.out_w_per_mask,
            self.in_w_per_mask,
            self.in_features // self.in_w_per_mask,
        )
        # back ot [768, 3072]
        masked_weight = masked_weight.reshape(self.out_features, self.in_features)

        out = torch.nn.functional.linear(input, masked_weight, self.bias)
        return out

    @classmethod
    def from_layer(cls, layer, out_w_per_mask, in_w_per_mask, mask_p, beta, layer_idx):
        assert type(layer) in [Conv1D, torch.nn.modules.linear.Linear]
        out_features, in_features = layer.weight.shape

        res = cls(
            mask_p=mask_p,
            beta=beta,
            layer_idx=layer_idx,
            in_features=in_features,
            out_features=out_features,
            bias=layer.bias is not None,
            out_w_per_mask=out_w_per_mask,
            in_w_per_mask=in_w_per_mask,
        )
        res.weight = layer.weight
        res.bias = layer.bias
        return res  # make sure to call cuda


def patch_hardconcrete(model, model_name, mask_p, beta):
    """
    out_w_per_mask: the number of output dims covered by a single mask parameter
    in_w_per_mask: the number of input dims covered by a single mask parameter
    ex: (1,1) for weight masking
        (768,1) for neuron masking
        (768, 768) for layer masking
    """
    out_w_per_mask = model.config.hidden_size
    in_w_per_mask = 1

    model.r_, model.l_, model.b_ = -0.1, 1.1, beta

    if "gpt2" in model_name:
        transpose_conv1d(model)

    # Replaces layers with their masked versions.
    for l in range(model.config.n_layer):
        attr_str = (
            f"{model.attr_dict['transformer_layer']}.{l}.{model.attr_dict['ffn_out']}"
        )
        ff_layer = get_attributes(model, attr_str)
        patch = MaskedLinear.from_layer(
            ff_layer, out_w_per_mask, in_w_per_mask, mask_p, beta, l
        )
        set_attributes(model, attr_str, patch)

    # shape should be [hidden_size, inner_dim]
    attr_str = f"{model.attr_dict['transformer_layer']}.{l}.{model.attr_dict['ffn_out']}.weight"
    shape = get_attributes(model, attr_str).shape
    assert shape[0] == model.config.hidden_size, shape


def reinit_hardconcrete(model, mask_p=None, beta=None):
    for l in range(model.config.n_layer):
        attr_str = f"{model.attr_dict['transformer_layer']}.{l}.{model.attr_dict['ffn_out']}.mask"
        mask_module = get_attributes(model, attr_str)
        if mask_p is not None:
            mask_module.mask_p = mask_p
        if beta is not None:
            mask_module.b = beta
        mask_module.init_weights()


def transpose_conv1d(model):
    for l in range(model.config.n_layer):
        attr_str = f"{model.attr_dict['transformer_layer']}.{l}.{model.attr_dict['ffn_out']}.weight"
        weight = get_attributes(model, attr_str)
        w_t = torch.nn.Parameter(weight.t())
        set_attributes(model, attr_str, w_t)


def compute_total_regularizer(model, start_layer_idx):
    total, n = 0, 0
    for module in model.modules():
        if hasattr(module, "regularizer"):
            if module.layer_idx >= start_layer_idx:
                total += module.regularizer()
                n += 1
    return total / n


@torch.no_grad()
def get_sparsity(model, threshold):
    total, n = 0, 0
    for l in range(model.config.n_layer):
        attr_str = f"{model.attr_dict['transformer_layer']}.{l}.{model.attr_dict['ffn_out']}.mask"
        module = get_attributes(model, attr_str)
        mask = module.produce_mask(is_train_runtime=False).squeeze()
        total += (mask < threshold).sum().item()
        n += len(mask)
    return total / n


def hard_concrete(lr, epoch, lambda_l1, stop_loss, threshold, model, inputs, gold_set):
    torch.manual_seed(0)
    model.eval()

    noise_dataloader = DataLoader(inputs, batch_size=64, shuffle=False)

    # start_layer_idx = args.start_mask_layer if hasattr(args, 'start_mask_layer') else 0
    start_layer_idx = 0

    # set tunable parameters
    print("Trainable Params:")
    cnt = 0
    params = []
    for n, p in model.named_parameters():
        if "mask_score" in n:
            cnt += 1
            if cnt > start_layer_idx:
                p.requires_grad = True
                print(n, p.shape)
            else:
                p.requires_grad = False
            params.append(p)
        else:
            p.requires_grad = False
    print("-" * 100)

    # training
    optimizer = torch.optim.Adam(params, lr=lr)
    model.zero_grad()
    scores, reg_losses, lm_losses = [], [], []
    for i in range(epoch):
        optimizer.zero_grad()

        for batch in noise_dataloader:

            outputs = model(batch, labels=batch)
            lm_loss = outputs.loss
            reg_loss = compute_total_regularizer(model, start_layer_idx)

            if (i + 1) % 10 == 0:
                sparsity = get_sparsity(model, threshold)
                print(
                    i + 1,
                    f"lm loss: {lm_loss.item():.3f}, reg_loss: {reg_loss.item():.3f}",
                )
                print("  Sparsity:", sparsity)

                ckpt_params = torch.sigmoid(
                    torch.stack(params).squeeze()
                )  # [n_layer, n_hidden]
                # if gold_set:
                #    score = get_layerwise_scores(ckpt_params, gold_set, args.ratio)
                # else:
                #    score = 0 # dummy
                #    if args.save_ckpt: save_params(args, ckpt_params, f'{i+1}.pt')
                # scores.append(score)
                lm_losses.append(lm_loss.item())
                reg_losses.append(reg_loss.item())
                if reg_loss < stop_loss:
                    break

            loss = lm_loss + lambda_l1 * reg_loss

            loss.backward()
            optimizer.step()

    params = torch.sigmoid(torch.stack(params)).detach().cpu()
    # params = torch.sigmoid(torch.stack(params).squeeze()).detach().cpu() # TODO this is the original
    # torch.save(params, os.path.join(args.out_dir, 'HC.pt'))
    # save_records(args, scores, np.array(reg_losses), np.array(lm_losses), sparsity)

    return params
