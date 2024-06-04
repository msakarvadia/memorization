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

from activations import register_hook
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
from transformers.pytorch_utils import Conv1D
import torch.nn.init as init

import random


torch.__version__
torch.manual_seed(0)
random.seed(0)

"""# Integrated Gradients"""


@torch.no_grad()
def get_ori_activations_IG(inner_dim, model, inputs):
    # seq_len = inputs['input_ids'].shape[1]
    seq_len = inputs.shape[1]
    ori_activations = torch.zeros((model.config.n_layer, seq_len, inner_dim))

    handles = []
    for ly in range(model.config.n_layer):
        handle = register_hook(
            model,
            ly,
            ori_activations,
            f"{model.attr_dict['transformer_layer']}.{ly}.{model.attr_dict['ffn_act']}",
        )
        handles.append(handle)

    out = model(inputs)

    for handle in handles:  # detach the hooks
        handle.remove()

    return ori_activations


def scaled_input(activations, steps, device):
    """
    Tiles activations along the batch dimension - gradually scaling them over
    `steps` steps from 0 to their original value over the batch dimensions.
    """
    tiled_activations = activations.expand((steps, len(activations)))
    scales = torch.linspace(start=0, end=1, steps=steps)[:, None]  # (steps, 1)
    out = (tiled_activations * scales).to(device)
    return out  # [steps, inner_dim]


def integrated_gradients(
    inner_dim, model, inputs, gold_set, ig_steps, device, n_batches=16, prompt_len=50
):
    activations = get_ori_activations_IG(inner_dim, model, inputs)

    target_ids = inputs.squeeze()[1:].tolist()
    seq_len = inputs.shape[1]

    n_layer = model.config.n_layer
    # prompt_start_i = args.prompt_len -1 if hasattr(args, 'prompt_len') else 0  # -1 for 0-indexed
    prompt_start_i = prompt_len - 1
    integrated_grads_ = torch.zeros((n_layer, seq_len - 1 - prompt_start_i, inner_dim))

    for ly in tqdm(range(n_layer)):
        integrated_grads = []
        for i in range(prompt_start_i, seq_len - 1):
            ori_activations = activations[ly, i]

            scaled_weights = scaled_input(
                ori_activations, steps=ig_steps, device=device
            )
            scaled_weights.requires_grad_(True)

            ff_attrs = f"{model.attr_dict['transformer_layer']}.{ly}.{model.attr_dict['ffn_act']}"
            integrated_grads_t = torch.zeros(inner_dim)
            for batch_weights in scaled_weights.chunk(n_batches):  # batch ig_steps
                bs = len(batch_weights)
                cur_input_ids = inputs[:, : i + 1].expand(
                    bs, i + 1
                )  # [ig_steps, cur_seq_len]

                # patch the model with the scaled activations
                patch_ff_layer(
                    model,
                    ff_attrs,
                    replacement_activations=batch_weights,
                )

                outputs = model(cur_input_ids)
                probs = torch.nn.functional.softmax(
                    outputs.logits[:, -1, :], dim=-1
                )  # [ig_steps, vocab]
                grad = torch.autograd.grad(
                    torch.unbind(probs[:, target_ids[i]]), batch_weights
                )[
                    0
                ]  # [ig_steps, inner_dim]
                integrated_grads_t += grad.sum(dim=0).cpu()  # sum over ig_steps

                unpatch_ff_layer(
                    model,
                    ff_attrs,
                )
            # Eq 5, 1/m * (ori - baseline) * (\Sum grads), where we use baseline = 0
            integrated_grads_t = ori_activations * integrated_grads_t / ig_steps
            integrated_grads.append(integrated_grads_t)

        integrated_grads_[ly] = torch.stack(integrated_grads, dim=0)

    # print(len(integrated_grads))
    ig_mean = integrated_grads_.mean(1).cpu()
    # torch.save(ig_mean, os.path.join(args.out_dir, 'ig-mean.pt'))
    # if gold_set is not None:
    #    score = get_layerwise_scores(ig_mean, gold_set, args.ratio)
    return ig_mean


def ig_full_data(
    inner_dim, model, inputs, gold_set, ig_steps, device, n_batches=16, prompt_len=50
):
    print("Inputs shape: ", inputs.shape)
    ig_mean = torch.zeros(model.config.n_layer, model.inner_dim)
    num_iters = 10
    for i, x in enumerate(inputs):
        ig_mean += integrated_gradients(
            inner_dim=model.inner_dim,
            model=model,
            inputs=x.unsqueeze(0),
            gold_set=None,
            ig_steps=20,
            device=device,
            n_batches=16,
        )

    ig_mean /= num_iters

    return ig_mean
