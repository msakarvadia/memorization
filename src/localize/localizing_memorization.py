import argparse
import pandas as pd
from src.data.old_data import *
import re

from src.localize.neuron.neuron_utils import (
    apply_ablation_mask_to_base_model,
    apply_ablation_mask_to_neurons,
    refined_check_percent_memorized,
    accuracy,
    perplexity,
    loss,
    compute_average_metric_accross_dataset,
    track_all_metrics,
    get_model,
)

from weight.greedy import do_greedy
from weight.durable import do_durable
from weight.obs import do_obs
from weight.greedy_obs import do_greedy_obs
from weight.greedy_obs2 import do_greedy_obs2
from weight.random_subnet import do_random
from weight.random_subnet_greedy import do_random_greedy

from neuron.zero_out import fast_zero_out_vector
from neuron.slimming import patch_slim, reinit_slim, compute_l1_loss, slim
from neuron.hard_concrete import (
    L0Mask,
    MaskedLinear,
    patch_hardconcrete,
    reinit_hardconcrete,
    transpose_conv1d,
    compute_total_regularizer,
    get_sparsity,
    hard_concrete,
)
from neuron.activations import register_hook, get_ori_activations_ACT, largest_act
from neuron.integrated_gradients import (
    ig_full_data,
    integrated_gradients,
    scaled_input,
    get_ori_activations_IG,
)

import torch
from torch.utils.data import DataLoader
import numpy as np
from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel

from tqdm import tqdm
import copy
import math
import time


device = "cuda" if torch.cuda.is_available() else "cpu"

from random import randrange, choices, sample
import random
from operator import add

from collections import OrderedDict
from typing import Dict, Callable
from transformers.pytorch_utils import Conv1D
import torch.nn.init as init
import random

print("finished imports")


def sort_metrics(
    args,
    perc_mem_dup_classes,
    perc_not_mem_dup_classes,
    perp_noise_dup_classes,
    perp_clean_dup_classes,
    accs_test,
    perplexities_test,
    total_time,
    # accBD,
    # percent_non_mem_bd,
    # perplex_BD_noise,
    # perplex_BD_clean,
):
    # Base dict
    data = vars(args)
    # we want localization method to be empty
    # this will also handle backdoors
    dup_dict = {
        "perc_mem_0": perc_mem_dup_classes[0],
        "perc_not_mem_correct_out_0": perc_not_mem_dup_classes[0],
        "perp_noise_0": perp_noise_dup_classes[0],
        "perp_clean_0": perp_clean_dup_classes[0],
    }
    time_dict = {
        "total_time": total_time,
    }
    if args.data_name in ["wiki_fast"]:
        data_dict = {
            "wiki_acc": [accs_test[0]],
            "wiki_perp": perplexities_test[0],
        }
        if args.duplicate and not args.backdoor:
            dup_dict = {
                "perc_mem_0": perc_mem_dup_classes[0],
                "perc_mem_1": perc_mem_dup_classes[1],
                "perc_mem_2": perc_mem_dup_classes[2],
                "perc_mem_3": perc_mem_dup_classes[3],
                "perc_not_mem_correct_out_0": perc_not_mem_dup_classes[0],
                "perc_not_mem_correct_out_1": perc_not_mem_dup_classes[1],
                "perc_not_mem_correct_out_2": perc_not_mem_dup_classes[2],
                "perc_not_mem_correct_out_3": perc_not_mem_dup_classes[3],
                "perp_noise_0": perp_noise_dup_classes[0],
                "perp_noise_1": perp_noise_dup_classes[3],
                "perp_noise_2": perp_noise_dup_classes[2],
                "perp_noise_3": perp_noise_dup_classes[3],
                "perp_clean_0": perp_clean_dup_classes[0],
                "perp_clean_1": perp_clean_dup_classes[1],
                "perp_clean_2": perp_clean_dup_classes[2],
                "perp_clean_3": perp_clean_dup_classes[3],
            }
        """
        # for wiki bd, we duplicate everything to the power of 2
        if args.duplicate and args.backdoor:
            dup_dict = {
                "perc_mem_2": perc_mem_dup_classes[0],
                "perc_not_mem_correct_out_2": perc_not_mem_dup_classes[0],
                "perp_noise_2": perp_noise_dup_classes[0],
                "perp_clean_2": perp_clean_dup_classes[0],
            }
        """
    if args.data_name in ["mult", "increment"]:
        data_dict = {
            "acc7": [accs_test[0]],
            "acc2": accs_test[1],
            "acc3": accs_test[2],
            "acc4": accs_test[3],
            "acc5": accs_test[4],
            "perp7": perplexities_test[0],
            "perp2": perplexities_test[1],
            "perp3": perplexities_test[2],
            "perp4": perplexities_test[3],
            "perp5": perplexities_test[4],
        }
        if args.duplicate:
            dup_dict = {
                "perc_mem_0": perc_mem_dup_classes[0],
                "perc_mem_1": perc_mem_dup_classes[1],
                "perc_mem_2": perc_mem_dup_classes[2],
                "perc_not_mem_correct_out_0": perc_not_mem_dup_classes[0],
                "perc_not_mem_correct_out_1": perc_not_mem_dup_classes[1],
                "perc_not_mem_correct_out_2": perc_not_mem_dup_classes[2],
                "perp_noise_0": perp_noise_dup_classes[0],
                "perp_noise_1": perp_noise_dup_classes[1],
                "perp_noise_2": perp_noise_dup_classes[2],
                "perp_clean_0": perp_clean_dup_classes[0],
                "perp_clean_1": perp_clean_dup_classes[1],
                "perp_clean_2": perp_clean_dup_classes[2],
            }

    data.update(data_dict)
    data.update(dup_dict)
    data.update(time_dict)
    return data


def check_basic_stats_existance(dict_of_values, df):
    """This is how we check if basic stats have been computed before"""
    exists = False
    model_path = dict_of_values["model_path"]
    if model_path in df["model_path"].unique():
        exists = True
    return exists


def check_existance(dict_of_values, df):
    """This is how we check if an experiment has been done before"""
    # https://stackoverflow.com/questions/24761133/pandas-check-if-row-exists-with-certain-values
    v = df.iloc[:, 0] == df.iloc[:, 0]
    for key, value in dict_of_values.items():
        v &= df[key] == value
    return v.any()


torch.__version__
torch.manual_seed(0)
random.seed(0)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results_path",
        type=str,
        default="weight_results.csv",
        help="Path to experiment results.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="../../../model_ckpts/mult_20000_3000_3000_3000_3000_20_150_0/four_layer/4_layer_6000_epoch.pth",
        help="Path to model ckpt file",
    )
    parser.add_argument(
        "--num_grads",
        type=int,
        default=512,
        help="Num of gradients to collect for OBS",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=512,
        help="Batch size (for checking memorization)",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=50,
        help="Blocksize for empirical fisher for OBS",
    )
    parser.add_argument(
        "--lambd",
        type=float,
        default=1e-7,
        help="Dampening factor for OBS",
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
        default="greedy_obs2",
        choices=[
            "greedy",
            "greedy_obs2",
            "durable",
            "durable_agg",
            "obs",
            "random",
            "random_greedy",
            "greedy_obs",
            "zero",
            "act",
            "ig",
            "slim",
            "hc",
        ],
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
        "--ig_steps",
        type=float,
        default=20,
        help="IG HP.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="Random/HP/SLIM HP: num epochs to optimize masks for",
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
        "--lr",
        type=float,
        default=0.1,
        help="Random/HC/SLIM HP: lr to optimize masks with",
    )
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Random HP: momentum to optimize masks with",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0005,
        help="Random HP: weight decay to optimize masks with",
    )
    parser.add_argument(
        "--backdoor",
        type=int,
        default=0,
        help="Whether or not to backdoor dataset.",
    )
    parser.add_argument(
        "--data_name",
        choices=[
            "wiki_fast",
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
    parser.add_argument(
        "--duplicate",
        type=int,
        default=0,
        help="Whether or not to do duplication on dataset.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=14,
        help="Number of tokens in model vocab.",
    )

    args = parser.parse_args()

    # Make the data
    print("Generating data...")
    data_path = f"../data/{args.data_name}_{args.num_7}_{args.num_2}_{args.num_3}_{args.num_4}_{args.num_5}_data_{args.length}_{args.num_test}_{args.num_noise}_{args.max_ctx}_{args.seed}.pt"
    pad_token_id = 13
    bos_token_id = 10
    eos_token_id = 11
    if args.data_name in ("shakespeare", "wiki", "wiki_fast"):
        data_path = f"../data/{args.data_name}_{args.max_ctx}_{args.seed}.pt"
        args.vocab_size = 50257
        tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        pad_token_id = tokenizer.eos_token_id
        eos_token_id = tokenizer.eos_token_id
        bos_token_id = tokenizer.bos_token_id
    if args.backdoor:
        print("Backdoor training")
        data_path = data_path[:-3]
        data_path = f"{data_path}_backdoor.pt"
    if args.duplicate:
        print("duplcate training")
        data_path = data_path[:-3]
        data_path = f"{data_path}_dup.pt"

    print("data path: ", data_path)
    print("Current working dir : %s" % os.getcwd())

    # We store locaization results in the parent dir of the edited models
    model_path, model_file_name = os.path.split(args.model_path)
    x = re.split("_", model_file_name)
    # print("Model epoch: ", x[2])
    model_path = model_path + "_edit/"
    args.results_path = f"{model_path}localization_results_{x[2]}.csv"
    if os.path.exists(args.results_path):
        print("checking if experiment stats are in resutls file")
        existing_results = pd.read_csv(args.results_path)
        data = vars(args)
        print(data)
        # need to check if "data" is in existing_results
        ckpt_check_df = existing_results[data.keys()]
        exists = check_existance(data, ckpt_check_df)
        print("This experiment exists: ", exists)
        if exists:
            exit()

    (
        noise_data,
        clean_data_corresponding_to_noise,
        train_datasets,
        clean_test_dataloaders,
        extra_train_datas,
        dup_idxs,
        trigger,
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
        backdoor=args.backdoor,
        duplicate=args.duplicate,
    )

    clean_data = train_datasets[1]

    # Get Model
    model = get_model(
        args.model_path, args.n_layers, args.max_ctx, args.n_embed, args.vocab_size
    )
    # TODO (MS): set this based on type of model
    model_name = "gpt2"

    print("BEFORE MASKING---------")
    total_time = (
        math.nan
    )  # sometime if neuron level attribs are computed, time will be na

    # only calculate new results if this if it isn't already in data
    # if os.path.exists(mem_seq_path):
    #    mem_seq = torch.load(mem_seq_path)
    #    print("Mem seq exists so base stats have to exist")
    #    exists = 1

    exists = 0
    if os.path.exists(args.results_path):
        print("checking if experiment stats are in resutls file")
        existing_results = pd.read_csv(args.results_path)
        data = vars(args)
        print(data)
        # need to check if "data" is in existing_results
        ckpt_check_df = existing_results[data.keys()]
        exists = check_basic_stats_existance(data, ckpt_check_df)
        print("The basic stats exists: ", exists)

    # make path for mem_seq and edited model
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    mem_seq_path = f"{model_path}mem_seq_{os.path.basename(args.model_path)}"

    # the base experiment exists so load it from the path
    if exists:
        mem_seq = torch.load(mem_seq_path)
    print("path for memorized sequences: ", mem_seq_path)

    base = 0
    if not exists:
        (
            perc_mem_dup_classes,
            perc_not_mem_dup_classes,
            perp_noise_dup_classes,
            perp_clean_dup_classes,
            mem_seq,
            clean_mem,
            accs_test,
            perplexities_test,
        ) = track_all_metrics(
            noise_data=noise_data,
            clean_data_corresponding_to_noise=clean_data_corresponding_to_noise,
            clean_test_dataloaders=clean_test_dataloaders,
            dup_idxs=dup_idxs,
            model=model,
            prompt_len=50,
            batch_size=args.batch_size,
            max_ctx=args.max_ctx,
            backdoor=args.backdoor,
            data_name=args.data_name,
            trigger=trigger,
        )
        # Save mem_seq in edited model_path
        torch.save(mem_seq, mem_seq_path)

        # there is no localization method for args
        base_args = copy.deepcopy(args)
        base_args.localization_method = "base_stats"

        data = sort_metrics(
            base_args,
            perc_mem_dup_classes,
            perc_not_mem_dup_classes,
            perp_noise_dup_classes,
            perp_clean_dup_classes,
            accs_test,
            perplexities_test,
            total_time,
        )

        base_df = pd.DataFrame.from_dict(data)
        base = 1

    if args.unlearn_set_name == "mem":
        print("unlearning memorized distribution")
        unlearn_set = mem_seq
        # print("Shape of unlearning set: ", unlearn_set.shape)
        if args.data_name == "wiki_fast":
            # only take subset of clean data as its too big
            shuffled_clean_data = random.shuffle(clean_data)
            extra_data = [
                clean_data[0:100000],
                noise_data,
            ]

        else:
            extra_data = [
                clean_data,
                noise_data,
                extra_train_datas[0],
                extra_train_datas[1],
                extra_train_datas[2],
                extra_train_datas[3],
            ]
    if args.unlearn_set_name == "noise":
        print("unlearning noise distribution")
        unlearn_set = noise_data
        if args.data_name == "wiki_fast":
            # only take subset of clean data as its too big
            shuffled_clean_data = random.shuffle(clean_data)
            extra_data = [
                clean_data[0:100000],
            ]

        else:
            extra_data = [
                clean_data,
                extra_train_datas[0],
                extra_train_datas[1],
                extra_train_datas[2],
                extra_train_datas[3],
            ]
    if args.unlearn_set_name == "seven":
        print("unlearning seven distribution")
        unlearn_set = clean_data
        extra_data = [
            noise_data,
            extra_train_datas[0],
            extra_train_datas[1],
            extra_train_datas[2],
            extra_train_datas[3],
        ]
    if args.unlearn_set_name == "two":
        print("unlearning two distribution")
        unlearn_set = extra_train_datas[0]
        extra_data = [
            clean_data,
            noise_data,
            extra_train_datas[1],
            extra_train_datas[2],
            extra_train_datas[3],
        ]
    if args.unlearn_set_name == "three":
        print("unlearning three distribution")
        unlearn_set = extra_train_datas[1]
        extra_data = [
            clean_data,
            noise_data,
            extra_train_datas[0],
            extra_train_datas[2],
            extra_train_datas[3],
        ]
    if args.unlearn_set_name == "four":
        print("unlearning four distribution")
        unlearn_set = extra_train_datas[2]
        extra_data = [
            clean_data,
            noise_data,
            extra_train_datas[0],
            extra_train_datas[1],
            extra_train_datas[3],
        ]
    if args.unlearn_set_name == "five":
        print("unlearning five distribution")
        unlearn_set = extra_train_datas[3]
        extra_data = [
            clean_data,
            noise_data,
            extra_train_datas[0],
            extra_train_datas[1],
            extra_train_datas[2],
        ]
    extra_data = torch.cat(extra_data, 0)

    if len(unlearn_set) != 0:
        # Check if procedure has already been done
        if args.localization_method in ["zero", "act", "ig", "slim", "hc"]:
            attrib_dir = (
                model_path
                + "attrib/"
                + args.localization_method
                + "/"
                + args.unlearn_set_name
                + "/"
            )
            if args.localization_method in ["hc", "slim"]:
                attrib_dir = (
                    attrib_dir
                    + f"{args.epochs}/{args.lambda_l1}/{args.stop_loss}/{args.lr}/"
                )
            if args.localization_method in ["ig"]:
                attrib_dir = attrib_dir + f"{args.ig_steps}/"
            name_of_attrib = attrib_dir + os.path.basename(args.model_path)
            # Make parent directories in path if it doesn't exist
            if not os.path.exists(attrib_dir):
                os.makedirs(attrib_dir)
            # If attrib file exists reload it
            if os.path.exists(name_of_attrib):
                print("Loading pre-computed attributions.")
                attributions = torch.load(name_of_attrib)
            # if it doesn't exist, create it
            else:

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

                    start = time.time()
                    attributions = hard_concrete(
                        lr=args.lr,
                        epoch=args.epochs,
                        lambda_l1=args.lambda_l1,
                        stop_loss=args.stop_loss,
                        threshold=1e-1,
                        model=model,
                        inputs=unlearn_set,
                        gold_set=None,
                        batch_size=args.batch_size,
                    )
                    end = time.time()
                    total_time = end - start

                ## Zero-out
                if args.localization_method == "zero":
                    start = time.time()
                    attributions = fast_zero_out_vector(
                        inner_dim=model.inner_dim,
                        n_batches=32,
                        model=model,
                        inputs=unlearn_set,
                        prompt_len=50,
                    )
                    end = time.time()
                    total_time = end - start
                ## Slimming
                if args.localization_method == "slim":
                    patched = False

                    if not patched:
                        patch_slim(model)
                        patched = True
                        model.to(device)  # send the coef_parameters in patch to gpu
                    else:
                        reinit_slim(model)
                    start = time.time()
                    attributions = slim(
                        lr=args.lr,
                        epoch=args.epochs,
                        lambda_l1=args.lambda_l1,
                        stop_loss=args.stop_loss,
                        threshold=1e-1,
                        model=model,
                        inputs=unlearn_set,
                        gold_set=None,
                        batch_size=args.batch_size,
                    )
                    end = time.time()
                    total_time = end - start

                ## Activations
                if args.localization_method == "act":

                    print("starting act localization")
                    start = time.time()
                    attributions = largest_act(
                        inner_dim=model.inner_dim,
                        model=model,
                        inputs=unlearn_set,
                        gold_set=None,
                        model_name=model_name,
                        prompt_len=50,
                        batch_size=args.batch_size,
                    )
                    end = time.time()
                    total_time = end - start

                ## Integrated Gradients
                if args.localization_method == "ig":

                    # attributions = integrated_gradients(
                    start = time.time()
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
                    end = time.time()
                    total_time = end - start
        if args.localization_method in ["ig", "slim", "hc", "zero", "act"]:
            print("Applying ablation mask to model")
            # this removes any patching and restores normal model
            # while still editing neurons by modifiying weights direction
            model = get_model(
                args.model_path,
                args.n_layers,
                args.max_ctx,
                args.n_embed,
                args.vocab_size,
            )
            model = apply_ablation_mask_to_base_model(
                attributions, model=model, ratio=args.ratio
            )

            # save the precomputed attributions
            torch.save(attributions, name_of_attrib)
        else:

            # WEIGHT LEVEL LOCALIZATION
            if args.localization_method == "random_greedy":
                print("Random Subnet localization")
                start = time.time()
                model = do_random_greedy(
                    model,
                    unlearn_set,
                    extra_data,
                    args.n_layers,
                    args.ratio,
                    args.epochs,
                    args.lr,
                    args.momentum,
                    args.weight_decay,
                    64,  # TODO make batch size an arg
                )
                end = time.time()
                total_time = end - start

            if args.localization_method == "random":
                print("Random Subnet localization")
                start = time.time()
                model = do_random(
                    model,
                    unlearn_set,
                    args.n_layers,
                    args.ratio,
                    args.epochs,
                    args.lr,
                    args.momentum,
                    args.weight_decay,
                )
                end = time.time()
                total_time = end - start
            sd = model.state_dict()
            original_model = get_model(
                args.model_path,
                args.n_layers,
                args.max_ctx,
                args.n_embed,
                args.vocab_size,
            )

            if args.localization_method == "greedy":
                print("Greedy localization")
                start = time.time()
                model = do_greedy(extra_data, unlearn_set, model, 64, args.ratio)
                end = time.time()
                total_time = end - start

            if args.localization_method == "obs":
                print("OBS localization")
                start = time.time()
                model = do_obs(
                    model,
                    unlearn_set,
                    args.ratio,
                    args.num_grads,
                    args.block_size,
                    args.lambd,
                )
                end = time.time()
                total_time = end - start

            if args.localization_method == "greedy_obs":
                print("Greedy OBS localization")
                start = time.time()
                model = do_greedy_obs(
                    model,
                    unlearn_set,
                    extra_data,
                    args.ratio,
                    args.num_grads,
                    args.block_size,
                    args.lambd,
                    64,
                )
                end = time.time()
                total_time = end - start

            if args.localization_method == "greedy_obs2":
                print("Greedy OBS localization V2")
                start = time.time()
                model = do_greedy_obs2(
                    model,
                    unlearn_set,
                    extra_data,
                    args.ratio,
                    args.num_grads,
                    args.block_size,
                    args.lambd,
                    64,
                )
                end = time.time()
                total_time = end - start

            if args.localization_method == "durable":
                print("Durable localization")
                start = time.time()
                model = do_durable(model, unlearn_set, args.ratio, False)
                end = time.time()
                total_time = end - start

            if args.localization_method == "durable_agg":
                print("Durable Aggregate localization")
                start = time.time()
                model = do_durable(model, unlearn_set, args.ratio, True)
                end = time.time()
                total_time = end - start

            original_sd = original_model.state_dict()
            # print("edit SD: ", sd.keys())
            # print("original SD: ", original_sd.keys())
            print(
                "Edited state dict matches original state dict: ",
                sd.keys() == original_sd.keys(),
            )

        print("Total time for unlearning (seconds): ", total_time)
        print("\n AFTER MASKING Ablation---------")

        # save model

        # have to save hyper-parameter specific model
        # this will work for act/zero/greedy/durable/durable_agg
        model_path = (
            model_path
            + args.localization_method
            + "/"
            + args.unlearn_set_name
            + "/"
            + str(args.ratio)
            + "/"
        )
        if args.localization_method in ["hc", "slim"]:
            model_path = (
                model_path
                + f"{args.epochs}/{args.lambda_l1}/{args.stop_loss}/{args.lr}/"
            )
        if args.localization_method in ["ig"]:
            model_path = model_path + f"{args.ig_steps}/"
        if args.localization_method in ["obs"]:
            model_path = (
                model_path + f"{args.block_size}/{args.num_grads}/{args.lambd}/"
            )
        if args.localization_method in ["random_greedy", "random"]:
            model_path = (
                model_path
                + f"{args.epochs}/{args.lr}/{args.momentum}/{args.weight_decay}/"
            )

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        MODEL_PATH = model_path + model_file_name

        print("MODEL PATH: ", MODEL_PATH)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
            },
            MODEL_PATH,
        )

        (
            perc_mem_dup_classes,
            perc_not_mem_dup_classes,
            perp_noise_dup_classes,
            perp_clean_dup_classes,
            mem_seq_all,
            clean_mem_seq_all,
            accs_test,
            perplexities_test,
        ) = track_all_metrics(
            noise_data=noise_data,
            clean_data_corresponding_to_noise=clean_data_corresponding_to_noise,
            clean_test_dataloaders=clean_test_dataloaders,
            dup_idxs=dup_idxs,
            model=model,
            prompt_len=50,
            batch_size=args.batch_size,
            max_ctx=args.max_ctx,
            backdoor=args.backdoor,
            data_name=args.data_name,
            trigger=trigger,
        )

        # save the memorized sequences after the edit
        mem_seq_path_post_edit = f"{model_path}mem_seq_{model_file_name}"
        print("path for the post edit mem_seq set: ", mem_seq_path_post_edit)
        torch.save(mem_seq_all, mem_seq_path_post_edit)

        data = sort_metrics(
            args,
            perc_mem_dup_classes,
            perc_not_mem_dup_classes,
            perp_noise_dup_classes,
            perp_clean_dup_classes,
            accs_test,
            perplexities_test,
            total_time,
        )
        ablate_df = pd.DataFrame.from_dict(data)

        # Now we concatentate all df together
        # if we already caluclated base_df, we don't reappend
        if base:
            print("appending experiment and base results")
            result = pd.concat([base_df, ablate_df], axis=0, ignore_index=True)
        if not base:
            print("appending only experiment not base results")
            result = pd.concat([ablate_df], axis=0, ignore_index=True)

        # Now open results.csv if it exisits and append
        if os.path.exists(args.results_path):
            print("appending to existing results file")
            existing_results = pd.read_csv(args.results_path)
            existing_results = pd.concat(
                [existing_results, result], axis=0, ignore_index=True
            )
            existing_results.to_csv(args.results_path, index=False)
        # Otherwise make a new results.csv
        else:
            print("making new results file")
            result.to_csv(args.results_path, index=False)

    # if we don't have anything in our mem seq, then we can still add our base_stats
    if len(unlearn_set) == 0:
        # Now we concatentate all df together
        # if we already caluclated base_df, we don't reappend
        print("result csv: ", args.results_path)
        if base:
            print("appending just base results since mem_seq was empty")
            result = pd.concat([base_df], axis=0, ignore_index=True)

            # Now open results.csv if it exisits and append
            if os.path.exists(args.results_path):
                print("appending to existing results file")
                existing_results = pd.read_csv(args.results_path)
                existing_results = pd.concat(
                    [existing_results, result], axis=0, ignore_index=True
                )
                existing_results.to_csv(args.results_path, index=False)
            # Otherwise make a new results.csv
            else:
                print("making new results file")
                result.to_csv(args.results_path, index=False)
