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

    (
        noise_data,
        clean_data_corresponding_to_noise,
        train_datasets,
        clean_test_dataloaders,
        extra_train_datas,
        dup_idxs,
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
    model_name = "gpt2"

    print("BEFORE MASKING---------")

    # print("shape of extra data: ", extra_train_datas[0].shape)
    (
        perc_mem_dup_classes,
        perc_not_mem_dup_classes,
        perp_noise_dup_classes,
        perp_clean_dup_classes,
        mem_seq,
        clean_mem,
        accs_test,
        perplexities_test,
        accBD,
        percent_non_mem_bd,
        perplex_BD_noise,
        perplex_BD_clean,
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
    )

    """
    data = {
        "model": [os.path.basename(args.model_path)],
        "localization_method": [""],
        "data_name": [args.data_name],
        "ablation_type": [""],
        "ratio": [""],
        "perc_mem": [perc_mem],
        # "acc": [acc],
        # "ppl_clean": [perplex_clean],
        # "ppl_noise": [perplex_noise],
        # "acc2": [acc2],
        # "acc3": [acc3],
        # "acc4": [acc4],
        # "acc5": [acc5],
        # "accbackdoor": [accBD],
        "seed": [args.seed],
        "num_grad": [args.num_grads],
        "block_size": [args.block_size],
        "lambd": [args.lambd],
        "unlearn_set": [args.unlearn_set_name],
    }
    """
    data = {}
    base_df = pd.DataFrame.from_dict(data)

    if args.unlearn_set_name == "mem":
        print("unlearning memorized distribution")
        unlearn_set = mem_seq
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
                "attrib/" + args.localization_method + "/" + args.unlearn_set_name + "/"
            )
            if args.localization_method in ["hc", "slim"]:
                attrib_dir = (
                    attrib_dir
                    + f"{args.epochs}/{args.lambda_l1}/{args.stop_loss}/{args.lr}/"
                )
            if args.localization_method in ["ig"]:
                attrib_dir = attrib_dir + f"{args.ig_steps}/"
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
                        epoch=args.epochs,
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
                        epoch=args.epochs,
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

                    print("starting act localization")
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
                # save the precomputed attributions
                torch.save(attributions, name_of_attrib)
        else:

            # WEIGHT LEVEL LOCALIZATION
            if args.localization_method == "random_greedy":
                print("Random Subnet localization")
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

            if args.localization_method == "random":
                print("Random Subnet localization")
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

            if args.localization_method == "greedy":
                print("Greedy localization")
                model = do_greedy(extra_data, unlearn_set, model, 64, args.ratio)

            if args.localization_method == "obs":
                print("OBS localization")
                model = do_obs(
                    model,
                    unlearn_set,
                    args.ratio,
                    args.num_grads,
                    args.block_size,
                    args.lambd,
                )

            if args.localization_method == "greedy_obs":
                print("Greedy OBS localization")
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

            if args.localization_method == "greedy_obs2":
                print("Greedy OBS localization V2")
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

            if args.localization_method == "durable":
                print("Durable localization")
                model = do_durable(model, unlearn_set, args.ratio, False)

            if args.localization_method == "durable_agg":
                print("Durable Aggregate localization")
                model = do_durable(model, unlearn_set, args.ratio, True)

        print("\n AFTER MASKING Ablation---------")

        # save model
        # MODEL_PATH = args.model_path[:-4] + f"_edit_{args.unlearn_set_name}.pth"
        model_path, model_file_name = os.path.split(args.model_path)
        model_path = model_path + "_edit/"
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
            accBD,
            percent_non_mem_bd,
            perplex_BD_noise,
            perplex_BD_clean,
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
        )
        """
        data = {
            "model": [os.path.basename(args.model_path)],
            "localization_method": [args.localization_method],
            "data_name": [args.data_name],
            "ablation_type": ["ablate"],
            "ratio": [args.ratio],
            "perc_mem": [perc_mem],
            # "acc": [acc],
            # "ppl_clean": [perplex_clean],
            # "ppl_noise": [perplex_noise],
            # "acc2": [acc2],
            # "acc3": [acc3],
            # "acc4": [acc4],
            # "acc5": [acc5],
            # "accbackdoor": [accBD],
            "seed": [args.seed],
            "num_grad": [args.num_grads],
            "block_size": [args.block_size],
            "lambd": [args.lambd],
            "unlearn_set": [args.unlearn_set_name],
        }
        """
        data = {}
        ablate_df = pd.DataFrame.from_dict(data)

        # Now we concatentate all df together
        result = pd.concat([base_df, ablate_df], axis=0, ignore_index=True)

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
