import argparse
import pandas as pd
from src.data.old_data import *

from src.localize.neuron.neuron_utils import (
    refined_check_percent_memorized,
    accuracy,
    perplexity,
    loss,
    compute_average_metric_accross_dataset,
    track_all_metrics,
    get_model,
)

from greedy import do_greedy
from durable import do_durable
from obs import do_obs
from random_subnet import do_random

import torch
from torch.utils.data import DataLoader
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
        default="../../../model_ckpts/mult_20000_3000_3000_3000_3000_20_150_0/four_layer/4_layer_6000_epoch.pth",
        help="Path to model ckpt file",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.00001,
        help="How many neurons to ablate",
    )
    parser.add_argument(
        "--localization_method",
        type=str,
        default="durable",
        choices=["greedy", "durable", "durable_agg", "obs", "random"],
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
        "--seed",
        type=int,
        default=0,
        help="Random seed for dataset generation.",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=4,
        help="Num layers in model",
    )
    parser.add_argument(
        "--data_name",
        choices=[
            "increment",
            "mult",
            # "exp",
            # "exponential",
            # "increment_3",
            # "mult_3",
            # "exp_3",
            # "exponential_3",
            # "increment_5",
            # "mult_5",
            # "exp_5",
            # "exponential_5",
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
            # "exponential_3",
            # "increment_5",
            # "mult_5",
            # "exp_5",
            # "exponential_5",
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

    # Get Model
    # TODO: add n_embed args
    model = get_model(args.model_path, args.n_layers, args.max_ctx, args.n_embed)
    model_name = "gpt2"

    """
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
    """
    if 0:
        # place holder
        print("hello")
    else:
        print("Recomputing attributions.")

        print("BEFORE MASKING---------")

        print("shape of extra data: ", extra_train_datas[0].shape)
        perc_mem, acc, perplex_clean, perplex_noise, mem_seq, clean_mem_seq = (
            track_all_metrics(
                noise_data=noise_data,
                clean_data_corresponding_to_noise=clean_data_corresponding_to_noise,
                clean_test_dataloaders=clean_test_dataloaders,
                model=model,
                prompt_len=50,
                batch_size=1000,
                max_ctx=args.max_ctx,
            )
        )
        if unlearn_set_name == "mem":
            unlearn_set = mem_seq
        if unlearn_set_name == "noise":
            unlearn_set = noise_data
        if unlearn_set_name == "seven":
            unlearn_set = clean_data
        if unlearn_set_name == "two":
            unlearn_set = extra_train_datas[0]
        if unlearn_set_name == "three":
            unlearn_set = extra_train_datas[1]
        if unlearn_set_name == "four":
            unlearn_set = extra_train_datas[2]
        if unlearn_set_name == "five":
            unlearn_set = extra_train_datas[3]

        if args.localization_method == "random":
            print("Random Subnet localization")
            model = do_random(model, unlearn_set, args.n_layers, args.ratio)

        if args.localization_method == "greedy":
            print("Greedy localization")
            clean_data = train_datasets[1]
            extra_data = [
                clean_data,
                extra_train_datas[1],
                extra_train_datas[2],
                extra_train_datas[3],
            ]
            extra_data = torch.cat(extra_data, 0)
            print(extra_data.shape)
            model = do_greedy(clean_data, unlearn_set, model, 64, args.ratio)
            # model = do_greedy(extra_data, extra_train_datas[0], model, 64, args.ratio)
            # model = do_greedy(clean_data, mem_seq, model, 64, args.ratio)
            # model = do_greedy(clean_data, noise_data, model, 64, args.ratio)
        if args.localization_method == "obs":
            print("OBS localization")
            # model = do_obs(model, mem_seq, args.ratio)
            model = do_obs(model, unlearn_set, args.ratio)

        if args.localization_method == "durable":
            print("Durable localization")
            model = do_durable(model, unlearn_set, args.ratio, False)
            # model = do_durable(model, mem_seq, args.ratio, False)
            # model = do_durable(model, noise_data, args.ratio, False)

        if args.localization_method == "durable_agg":
            print("Durable Aggregate localization")
            model = do_durable(model, unlearn_set, args.ratio, True)
            # model = do_durable(model, mem_seq, args.ratio, True)
            # model = do_durable(model, noise_data, args.ratio, True)

        print("\n AFTER MASKING Ablation---------")

        perc_mem, acc, perplex_clean, perplex_noise, mem_seq, clean_mem_seq = (
            track_all_metrics(
                noise_data=noise_data,
                clean_data_corresponding_to_noise=clean_data_corresponding_to_noise,
                clean_test_dataloaders=clean_test_dataloaders,
                model=model,
                prompt_len=50,
                batch_size=1000,
                max_ctx=args.max_ctx,
            )
        )

        """
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
    }
    base_df = pd.DataFrame.from_dict(data)
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
    result = pd.concat([base_df, noise_df, ablate_df, mean_df])

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
    """
