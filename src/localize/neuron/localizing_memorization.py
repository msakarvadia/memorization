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
    ig_full_data,
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
        "--results_path",
        type=str,
        default="neuron_results.csv",
        help="Path to experiment results.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Path to model ckpt file",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.01,
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
        "--max_ctx",
        type=int,
        default=150,
        help="Size of maximum context",
    )
    parser.add_argument(
        "--n_embed",
        type=int,
        default=128,
        help="Embbed dim of model (size of hidden states).",
    )
    parser.add_argument(
        "--num_7",
        type=int,
        default=20000,
        help="Number of points from the 7 distribution.",
    )
    parser.add_argument(
        "--num_2",
        type=int,
        default=3000,
        help="Number of points from the 2 distribution.",
    )
    parser.add_argument(
        "--num_3",
        type=int,
        default=3000,
        help="Number of points from the 3 distribution.",
    )
    parser.add_argument(
        "--num_4",
        type=int,
        default=3000,
        help="Number of points from the 4 distribution.",
    )
    parser.add_argument(
        "--num_5",
        type=int,
        default=3000,
        help="Number of points from the 5 distribution.",
    )
    parser.add_argument(
        "--num_noise",
        type=int,
        default=1000,
        help="Number of points from the 7 distribution to use in noise set.",
    )
    parser.add_argument(
        "--num_test",
        type=int,
        default=1000,
        help="Number of points from each distribution to use in test set.",
    )
    parser.add_argument(
        "--length",
        type=int,
        default=20,
        help="Amount of numbers in each math sequence",
    )
    parser.add_argument(
        "--lambda_l1",
        type=float,
        default=1000,
        help="HC/Slim HP.",
    )
    parser.add_argument(
        "--stop_loss",
        type=float,
        default=1e-1,
        help="HC/Slim HP.",
    )
    parser.add_argument(
        "--epoch",
        type=int,
        default=100,
        help="HC/Slim HP.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-2,
        help="HC/Slim HP.",
    )
    parser.add_argument(
        "--ig_steps",
        type=float,
        default=20,
        help="IG HP.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for dataset generation.",
    )
    parser.add_argument(
        "--data_name",
        choices=[
            "increment",
            "mult",
        ],
        type=str,
        default="mult",
        help="Name of function type you want to train with.",
    )
    parser.add_argument(
        "--unlearn_set_name",
        choices=[
            "mem",
            "noise",
            "seven",
            "two",
            "three",
            "four",
            "five",
        ],
        type=str,
        default="mem",
        help="Name of dataset you want to unlearn.",
    )

    args = parser.parse_args()

    # Make the data
    print("Generating data...")
    data_path = f"../../data/{args.data_name}_{args.num_7}_{args.num_2}_{args.num_3}_{args.num_4}_{args.num_5}_data_{args.length}_{args.num_test}_{args.num_noise}_{args.max_ctx}_{args.seed}.pt"

    (
        noise_data,
        clean_data_corresponding_to_noise,
        train_datasets,
        clean_test_dataloaders,
        extra_train_datas,
    ) = get_data(
        data_name=args.data_name,
        num_7=args.num_7,
        num_2=args.num_2,
        num_3=args.num_3,
        num_4=args.num_4,
        num_5=args.num_5,
        num_noise=args.num_noise,
        num_test=args.num_test,
        data_path_name=data_path,
        length=args.length,
        seed=args.seed,
        max_ctx=args.max_ctx,
    )
    clean_data = train_datasets[1]

    # Get Model
    model = get_model(args.model_path, args.n_layers, args.max_ctx, args.n_embed)
    model_name = "gpt2"

    print("BEFORE MASKING---------")

    (
        perc_mem,
        acc,
        perplex_clean,
        perplex_noise,
        mem_seq,
        clean_mem_seq,
        acc2,
        acc3,
        acc4,
        acc5,
    ) = track_all_metrics(
        noise_data=noise_data,
        clean_data_corresponding_to_noise=clean_data_corresponding_to_noise,
        clean_test_dataloaders=clean_test_dataloaders,
        model=model,
        prompt_len=50,
        batch_size=1000,
        max_ctx=args.max_ctx,
    )

    data = {
        "model": [os.path.basename(args.model_path)],
        "localization_method": [""],
        "data_name": [args.data_name],
        "ablation_type": [""],
        "ratio": [""],
        "perc_mem": [perc_mem],
        "acc": [acc],
        "ppl_clean": [perplex_clean],
        "ppl_noise": [perplex_noise],
        "acc2": [acc2],
        "acc3": [acc3],
        "acc4": [acc4],
        "acc5": [acc5],
        "seed": [args.seed],
        "lambda_l1": [args.lambda_l1],
        "stop_loss": [args.stop_loss],
        "epoch": [args.epoch],
        "lr": [args.lr],
        "ig_steps": [args.ig_steps],
        "unlearn_set": [args.unlearn_set_name],
    }
    base_df = pd.DataFrame.from_dict(data)

    print("Recomputing attributions.")
    if args.unlearn_set_name == "mem":
        print("unlearning memorized distribution")
        unlearn_set = mem_seq
    if args.unlearn_set_name == "noise":
        print("unlearning noise distribution")
        unlearn_set = noise_data
    if args.unlearn_set_name == "seven":
        print("unlearning seven distribution")
        unlearn_set = clean_data
    if args.unlearn_set_name == "two":
        print("unlearning two distribution")
        unlearn_set = extra_train_datas[0]
    if args.unlearn_set_name == "three":
        print("unlearning three distribution")
        unlearn_set = extra_train_datas[1]
    if args.unlearn_set_name == "four":
        print("unlearning four distribution")
        unlearn_set = extra_train_datas[2]
    if args.unlearn_set_name == "five":
        print("unlearning five distribution")
        unlearn_set = extra_train_datas[3]

    # Check if procedure has already been done
    attrib_dir = (
        "attrib/" + args.localization_method + "/" + args.unlearn_set_name + "/"
    )
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

        if len(unlearn_set) != 0:

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
                    lr=args.lr,
                    epoch=args.epoch,
                    lambda_l1=args.lambda_l1,
                    stop_loss=args.stop_loss,
                    threshold=1e-1,
                    model=model,
                    inputs=unlearn_set,
                    gold_set=None,
                )

            ## Zero-out
            if args.localization_method == "zero":
                attributions = fast_zero_out_vector(
                    inner_dim=model.inner_dim,
                    n_batches=16,
                    model=model,
                    inputs=unlearn_set,
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
                    lr=args.lr,
                    epoch=args.epoch,
                    lambda_l1=args.lambda_l1,
                    stop_loss=args.stop_loss,
                    threshold=1e-1,
                    model=model,
                    inputs=unlearn_set,
                    # inputs=mem_seq,
                    # inputs=noise_data,
                    gold_set=None,
                )

            ## Activations
            if args.localization_method == "act":

                attributions = largest_act(
                    inner_dim=model.inner_dim,
                    model=model,
                    # inputs=noise_data,
                    inputs=unlearn_set,
                    # inputs=mem_seq,
                    gold_set=None,
                    model_name="gpt2",
                    prompt_len=50,
                )

            ## Integrated Gradients
            if args.localization_method == "ig":

                # attributions = integrated_gradients(
                attributions = ig_full_data(
                    inner_dim=model.inner_dim,
                    model=model,
                    # inputs=noise_data[0].unsqueeze(0),
                    inputs=mem_seq,
                    gold_set=None,
                    ig_steps=args.ig_steps,
                    device=device,
                    n_batches=16,
                    prompt_len=50,
                )

            # now save those attributions
            torch.save(attributions, name_of_attrib)

    # Only evaluate if there were items in unlearning set
    if len(unlearn_set) != 0:

        ## evaluate localization strategies
        model = get_model(args.model_path, args.n_layers, args.max_ctx, args.n_embed)

        apply_ablation_mask_to_neurons(attributions, model=model, ratio=args.ratio)

        print("\n AFTER MASKING Ablation---------")
        (
            perc_mem,
            acc,
            perplex_clean,
            perplex_noise,
            mem_seq,
            clean_mem_seq,
            acc2,
            acc3,
            acc4,
            acc5,
        ) = track_all_metrics(
            noise_data=noise_data,
            clean_data_corresponding_to_noise=clean_data_corresponding_to_noise,
            clean_test_dataloaders=clean_test_dataloaders,
            model=model,
            prompt_len=50,
            batch_size=1000,
            max_ctx=args.max_ctx,
        )
        data = {
            "model": [os.path.basename(args.model_path)],
            "localization_method": [args.localization_method],
            "data_name": [args.data_name],
            "ablation_type": [""],
            "ratio": [""],
            "perc_mem": [perc_mem],
            "acc": [acc],
            "ppl_clean": [perplex_clean],
            "ppl_noise": [perplex_noise],
            "acc2": [acc2],
            "acc3": [acc3],
            "acc4": [acc4],
            "acc5": [acc5],
            "seed": [args.seed],
            "lambda_l1": [args.lambda_l1],
            "stop_loss": [args.stop_loss],
            "epoch": [args.epoch],
            "lr": [args.lr],
            "ig_steps": [args.ig_steps],
            "unlearn_set": [args.unlearn_set_name],
        }

        ablate_df = pd.DataFrame.from_dict(data)

        remove_ablation_mask_from_neurons(model)

        ##################

        apply_mean_ablation_mask_to_neurons(
            attributions, model=model, inputs=noise_data, ratio=args.ratio
        )

        # remove_ablation_mask_from_neurons(model)
        print("\n AFTER MASKING Mean---------")

        (
            perc_mem,
            acc,
            perplex_clean,
            perplex_noise,
            mem_seq,
            clean_mem_seq,
            acc2,
            acc3,
            acc4,
            acc5,
        ) = track_all_metrics(
            noise_data=noise_data,
            clean_data_corresponding_to_noise=clean_data_corresponding_to_noise,
            clean_test_dataloaders=clean_test_dataloaders,
            model=model,
            prompt_len=50,
            batch_size=1000,
            max_ctx=args.max_ctx,
        )

        data = {
            "model": [os.path.basename(args.model_path)],
            "localization_method": [args.localization_method],
            "data_name": [args.data_name],
            "ablation_type": [""],
            "ratio": [""],
            "perc_mem": [perc_mem],
            "acc": [acc],
            "ppl_clean": [perplex_clean],
            "ppl_noise": [perplex_noise],
            "acc2": [acc2],
            "acc3": [acc3],
            "acc4": [acc4],
            "acc5": [acc5],
            "seed": [args.seed],
            "lambda_l1": [args.lambda_l1],
            "stop_loss": [args.stop_loss],
            "epoch": [args.epoch],
            "lr": [args.lr],
            "ig_steps": [args.ig_steps],
            "unlearn_set": [args.unlearn_set_name],
        }
        mean_df = pd.DataFrame.from_dict(data)

        remove_ablation_mask_from_neurons(model)

        ##################

        apply_noise_ablation_mask_to_neurons(
            attributions, model=model, inputs=noise_data, ratio=args.ratio
        )

        print("\n AFTER MASKING Noise---------")

        (
            perc_mem,
            acc,
            perplex_clean,
            perplex_noise,
            mem_seq,
            clean_mem_seq,
            acc2,
            acc3,
            acc4,
            acc5,
        ) = track_all_metrics(
            noise_data=noise_data,
            clean_data_corresponding_to_noise=clean_data_corresponding_to_noise,
            clean_test_dataloaders=clean_test_dataloaders,
            model=model,
            prompt_len=50,
            batch_size=1000,
            max_ctx=args.max_ctx,
        )

        data = {
            "model": [os.path.basename(args.model_path)],
            "localization_method": [args.localization_method],
            "data_name": [args.data_name],
            "ablation_type": [""],
            "ratio": [""],
            "perc_mem": [perc_mem],
            "acc": [acc],
            "ppl_clean": [perplex_clean],
            "ppl_noise": [perplex_noise],
            "acc2": [acc2],
            "acc3": [acc3],
            "acc4": [acc4],
            "acc5": [acc5],
            "seed": [args.seed],
            "lambda_l1": [args.lambda_l1],
            "stop_loss": [args.stop_loss],
            "epoch": [args.epoch],
            "lr": [args.lr],
            "ig_steps": [args.ig_steps],
            "unlearn_set": [args.unlearn_set_name],
        }
        noise_df = pd.DataFrame.from_dict(data)

        remove_ablation_mask_from_neurons(model)

        # Now we concatentate all df together
        result = pd.concat([base_df, noise_df, ablate_df, mean_df])

        # Now open results.csv if it exisits and append
        if os.path.exists(args.results_path):
            print("appending to existing results file")
            existing_results = pd.read_csv(args.results_path)
            existing_results = pd.concat([existing_results, result])
            existing_results.to_csv(args.results_path, index=False)
        # Otherwise make a new results.csv
        else:
            print("making new results file")
            result.to_csv(args.results_path, index=False)
