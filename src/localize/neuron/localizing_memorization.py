import sys
import argparse
import pandas as pd
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
    remove_all_forward_hooks,
)
from src.data.old_data import *

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
from activations import register_hook, get_ori_activations_ACT, largest_act
from integrated_gradients import (
    integrated_gradients,
    scaled_input,
    get_ori_activations_IG,
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
from transformers.pytorch_utils import Conv1D
import torch.nn.init as init

import random

print("finished imports")

torch.__version__
torch.manual_seed(0)
random.seed(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model ckpt file",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.1,
        help="How many neurons to ablate",
    )
    parser.add_argument(
        "--localization_method",
        type=str,
        default="zero",
        choices=["zero", "act", "ig", "slim", "hc"],
        help="Path to model ckpt file",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=1,
        help="Path to model ckpt file",
    )
    parser.add_argument(
        "--data_name",
        choices=["increment", "mult", "exp", "exponential"],
        type=str,
        default="increment",
        help="Name of function type you want to train with.",
    )

    args = parser.parse_args()

    # Make the data
    print("Generating data...")
    data_path = f"../../data/{args.data_name}_data.pt"

    (
        noise_data,
        clean_data_corresponding_to_noise,
        train_datasets,
        clean_test_dataloaders,
    ) = get_data(data_name=args.data_name, num_test=1000, data_path_name=data_path)

    # Get Model
    model = get_model(args.model_path, args.n_layers)
    model_name = "gpt2"

    # Check if procedure has already been done
    attrib_dir = "attrib/" + args.localization_method + "/"
    name_of_attrib = attrib_dir + os.path.basename(args.model_path)
    print(name_of_attrib)
    # Make parent directories in path if it doesn't exist
    if not os.path.exists(attrib_dir):
        os.makedirs(attrib_dir)
    # If attrib file exists reload it
    if os.path.exists(name_of_attrib):
        print("Loading pre-computed attributions.")
        attributions = torch.load(name_of_attrib)
    # if it doesn't exist, create it
    else:
        print("Recomputing attributions.")

        ## Hard concrete
        if args.localization_method == "hc":
            patched = False

            if not patched:
                patch_hardconcrete(model, model_name, mask_p=0.5, beta=2 / 3)
                patched = True
                model.to(device)
            else:
                if (
                    "gpt2" in model_name
                ):  # the newly loaded weights need to be transposed
                    transpose_conv1d(model)
                reinit_hardconcrete(model)

            attributions = hard_concrete(
                lr=1e-2,
                epoch=100,
                lambda_l1=1000,
                stop_loss=1e-1,
                threshold=1e-1,
                model=model,
                inputs=noise_data,
                gold_set=None,
            )

        ## Zero-out
        if args.localization_method == "zero":
            attributions = fast_zero_out_vector(
                inner_dim=model.inner_dim,
                n_batches=16,
                model=model,
                inputs=noise_data,
                labels=clean_data_corresponding_to_noise,
                prompt_len=50,
            )

        ## Slimming
        if args.localization_method == "slim":
            patched = False

            if not patched:
                patch_slim(model)
                patched = True
                model.to(device)  # send the coef_parameters in patch to gpu
            else:
                reinit_slim(model)
            attributions = slim(
                lr=1e-2,
                epoch=100,
                lambda_l1=1000,
                stop_loss=1e-1,
                threshold=1e-1,
                model=model,
                inputs=noise_data,
                gold_set=None,
            )

        ## Activations
        if args.localization_method == "act":

            attributions = largest_act(
                inner_dim=model.inner_dim,
                model=model,
                inputs=noise_data,
                gold_set=None,
                model_name="gpt2",
                prompt_len=50,
            )

        ## Integrated Gradients
        if args.localization_method == "ig":

            attributions = integrated_gradients(
                inner_dim=model.inner_dim,
                model=model,
                inputs=noise_data[0].unsqueeze(0),
                gold_set=None,
                ig_steps=200,
                device=device,
                n_batches=16,
                prompt_len=50,
            )

        # now save those attributions
        torch.save(attributions, name_of_attrib)

    ## evaluate localization strategies
    model = get_model(args.model_path, args.n_layers)

    print("BEFORE MASKING---------")

    perc_mem, acc, perplex_clean, perplex_noise = track_all_metrics(
        noise_data=noise_data,
        clean_data_corresponding_to_noise=clean_data_corresponding_to_noise,
        clean_test_dataloaders=clean_test_dataloaders,
        model=model,
        prompt_len=50,
        batch_size=1000,
    )
    ##################

    apply_ablation_mask_to_neurons(attributions, model=model, ratio=args.ratio)

    print("\n AFTER MASKING Ablation---------")

    perc_mem, acc, perplex_clean, perplex_noise = track_all_metrics(
        noise_data=noise_data,
        clean_data_corresponding_to_noise=clean_data_corresponding_to_noise,
        clean_test_dataloaders=clean_test_dataloaders,
        model=model,
        prompt_len=50,
        batch_size=1000,
    )

    data = {
        "model": [os.path.basename(args.model_path)],
        "localization_method": [args.localization_method],
        "data_name": [args.data_name],
        "ablation_type": ["ablate"],
        "ratio": [args.ratio],
        "perc_mem": [perc_mem],
        "acc": [acc],
        "ppl_clean": [perplex_clean],
        "ppl_noise": [perplex_noise],
    }
    ablate_df = pd.DataFrame.from_dict(data)

    remove_ablation_mask_from_neurons(model)

    ##################

    apply_mean_ablation_mask_to_neurons(
        attributions, model=model, inputs=noise_data, ratio=args.ratio
    )

    # remove_ablation_mask_from_neurons(model)
    print("\n AFTER MASKING Mean---------")

    perc_mem, acc, perplex_clean, perplex_noise = track_all_metrics(
        noise_data=noise_data,
        clean_data_corresponding_to_noise=clean_data_corresponding_to_noise,
        clean_test_dataloaders=clean_test_dataloaders,
        model=model,
        prompt_len=50,
        batch_size=1000,
    )

    data = {
        "model": [os.path.basename(args.model_path)],
        "localization_method": [args.localization_method],
        "data_name": [args.data_name],
        "ablation_type": ["mean"],
        "ratio": [args.ratio],
        "perc_mem": [perc_mem],
        "acc": [acc],
        "ppl_clean": [perplex_clean],
        "ppl_noise": [perplex_noise],
    }
    mean_df = pd.DataFrame.from_dict(data)

    remove_ablation_mask_from_neurons(model)

    ##################

    apply_noise_ablation_mask_to_neurons(
        attributions, model=model, inputs=noise_data, ratio=args.ratio
    )

    print("\n AFTER MASKING Noise---------")

    perc_mem, acc, perplex_clean, perplex_noise = track_all_metrics(
        noise_data=noise_data,
        clean_data_corresponding_to_noise=clean_data_corresponding_to_noise,
        clean_test_dataloaders=clean_test_dataloaders,
        model=model,
        prompt_len=50,
        batch_size=1000,
    )

    data = {
        "model": [os.path.basename(args.model_path)],
        "localization_method": [args.localization_method],
        "data_name": [args.data_name],
        "ablation_type": ["noise"],
        "ratio": [args.ratio],
        "perc_mem": [perc_mem],
        "acc": [acc],
        "ppl_clean": [perplex_clean],
        "ppl_noise": [perplex_noise],
    }
    noise_df = pd.DataFrame.from_dict(data)

    remove_ablation_mask_from_neurons(model)

    # Now we concatentate all df together
    result = pd.concat([noise_df, ablate_df, mean_df])

    results_path = "neuron_results.csv"
    # Now open results.csv if it exisits and append
    if os.path.exists(results_path):
        print("appending to existing results file")
        existing_results = pd.read_csv(results_path)
        existing_results = pd.concat([existing_results, result])
        existing_results.to_csv(results_path, index=False)
    # Otherwise make a new results.csv
    else:
        print("making new results file")
        result.to_csv(results_path, index=False)
