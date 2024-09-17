""" This script calls `localize_memorization` and does a HP sweep over all of the methods, as an arugment, this script will only take info about the model path, model cfg, and data settings"""

import os
import argparse


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default="../../model_ckpts/lm_test/wiki_4_noise_dup/4_layer_30_epoch.pth",
        help="Path to model ckpt file",
    )
    parser.add_argument(
        "--num_extra",
        type=int,
        default=3000,
        help="Number of extra points from the 2/3/4/5 distribution.",
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
        "--backdoor",
        type=int,
        default=0,
        help="Whether or not to backdoor dataset.",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="",
        choices=[
            "EleutherAI/pythia-2.8b-deduped",
            "EleutherAI/pythia-6.9b-deduped",
        ],
        help="name of model",
    )
    parser.add_argument(
        "--step",
        type=int,
        default=143000,
        choices=[36000, 72000, 108000, 143000],
        help="The version of the model we load.",
    )
    parser.add_argument(
        "--data_name",
        choices=[
            "wiki_fast",
            "increment",
            "mult",
        ],
        type=str,
        default="wiki_fast",
        help="Name of function type you want to train with.",
    )
    parser.add_argument(
        "--duplicate",
        type=int,
        default=1,
        help="Whether or not to do duplication on dataset.",
    )

    args = parser.parse_args()
    for loc_method in [
        "obs",
        "ig",
        "greedy",
        "random_greedy",
        "zero",
        "act",
        "hc",
        "slim",
        "durable",
        "durable_agg",
        "random",
    ]:
        # TODO (MS): add in more ratios
        for ratio in [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.25, 0.3, 0.5, 0.8]:
            # do not do "ig" for 16 layer models
            if "16" in args.model_path:
                if loc_method in ["ig"]:
                    continue
                if loc_method in ["greedy"]:
                    if ratio >= 0.05:
                        continue

            if "wiki" in args.model_path:
                if loc_method in ["greedy"]:
                    if ratio >= 0.05:
                        continue

            # want to reserve high ratios for random based methods
            if loc_method not in ["random", "random_greedy"]:
                if ratio >= 0.1:
                    continue

            # this ratio is too small for neuron-level methods
            if loc_method in ["zero", "hc", "ig", "slim", "act"]:
                if ratio <= 0.0001:
                    continue

            if loc_method in ["greedy"]:
                if ratio > 0.05:
                    continue
                if args.model_name != "":
                    # NOTE (MS): for pythia models its too slow to iterate thru each individual weight for greedy
                    if ratio > 0.00001:
                        continue

            if loc_method in ["ig"]:
                for ig_steps in [20]:
                    if args.model_name == "":
                        command = f"""python localizing_memorization.py\
                                 --model_path {args.model_path}\
                                --n_layer {args.n_layers}\
                                --seed {args.seed}\
                                --duplicate {args.duplicate}\
                                --backdoor {args.backdoor}\
                                --data_name {args.data_name}\
                                --num_2 {args.num_extra}\
                                --num_3 {args.num_extra}\
                                --num_4 {args.num_extra}\
                                --num_5 {args.num_extra}\
                                --length 20\
                                --max_ctx 150\
                                --batch_size 32\
                                --ratio {ratio}\
                                --ig_steps {ig_steps}\
                                --localization_method {loc_method}"""
                    else:
                        continue
                    os.system(command)
                    print("RAN COMMAND")
            if loc_method in ["slim", "hc", "random"]:
                for epochs in [1, 10, 20]:
                    if args.model_name == "":
                        command = f"""python localizing_memorization.py\
                                 --model_path {args.model_path}\
                                --n_layer {args.n_layers}\
                                --seed {args.seed}\
                                --duplicate {args.duplicate}\
                                --backdoor {args.backdoor}\
                                --data_name {args.data_name}\
                                --num_2 {args.num_extra}\
                                --num_3 {args.num_extra}\
                                --num_4 {args.num_extra}\
                                --num_5 {args.num_extra}\
                                --length 20\
                                --max_ctx 150\
                                --batch_size 32\
                                --ratio {ratio}\
                                --epochs {epochs}\
                                --localization_method {loc_method}"""
                    else:
                        command = f"""python prod_grade.py\
                                --model_name {args.model_name}\
                                --step {args.step}\
                                --seed {args.seed}\
                                --batch_size 32\
                                --ratio {ratio}\
                                --epochs {epochs}\
                                --localization_method {loc_method}"""
                    os.system(command)
                    print("RAN COMMAND")
            if loc_method in ["random_greedy"]:
                for loss_weight in [0.9, 0.7, 0.5]:
                    for epochs in [1, 10, 20]:
                        if args.model_name == "":
                            command = f"""python localizing_memorization.py\
                                     --model_path {args.model_path}\
                                    --n_layer {args.n_layers}\
                                    --seed {args.seed}\
                                    --duplicate {args.duplicate}\
                                    --backdoor {args.backdoor}\
                                    --data_name {args.data_name}\
                                    --num_2 {args.num_extra}\
                                    --num_3 {args.num_extra}\
                                    --num_4 {args.num_extra}\
                                    --num_5 {args.num_extra}\
                                    --length 20\
                                    --max_ctx 150\
                                    --batch_size 32\
                                    --ratio {ratio}\
                                    --epochs {epochs}\
                                    --loss_weighting {loss_weight}\
                                    --localization_method {loc_method}"""
                        else:
                            command = f"""python prod_grade.py\
                                    --model_name {args.model_name}\
                                    --step {args.step}\
                                    --seed {args.seed}\
                                    --batch_size 32\
                                    --ratio {ratio}\
                                    --epochs {epochs}\
                                    --loss_weighting {loss_weight}\
                                    --localization_method {loc_method}"""
                        os.system(command)
                        print("RAN COMMAND")
            # else:
            if loc_method in [
                "zero",
                "act",
                "durable",
                "durable_agg",
                "greedy",
                "obs",
                "greedy",
            ]:
                if args.model_name == "":
                    command = f"""python localizing_memorization.py\
                             --model_path {args.model_path}\
                            --n_layer {args.n_layers}\
                            --seed {args.seed}\
                            --duplicate {args.duplicate}\
                            --backdoor {args.backdoor}\
                            --data_name {args.data_name}\
                            --num_2 {args.num_extra}\
                            --num_3 {args.num_extra}\
                            --num_4 {args.num_extra}\
                            --num_5 {args.num_extra}\
                            --length 20\
                            --max_ctx 150\
                            --ratio {ratio}\
                            --localization_method {loc_method}"""
                else:
                    if loc_method in ["obs", "zero"]:
                        continue
                    command = f"""python prod_grade.py\
                            --model_name {args.model_name}\
                            --step {args.step}\
                            --seed {args.seed}\
                            --batch_size 32\
                            --ratio {ratio}\
                            --localization_method {loc_method}"""
                os.system(command)
                print("RAN COMMAND")
