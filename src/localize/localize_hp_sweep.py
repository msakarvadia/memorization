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
        "zero",
        "act",
        "hc",
        "slim",
        "ig",
        "greedy",
        "durable",
        "durable_agg",
        "obs",
        "random",
        "random_greedy",
    ]:
        # TODO (MS): add in more ratios
        for ratio in [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.25, 0.50, 0.75, 0.9]:
            # want to reserve high ratios for random based methods
            if loc_method not in ["random", "random_greedy"]:
                if ratio >= 0.1:
                    continue

            if loc_method in ["ig"]:
                for ig_steps in [1, 10, 20]:
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
                    os.system(command)
                    print("RAN COMMAND")
            if loc_method in ["slim", "hc", "greedy", "random", "random_greedy"]:
                for epochs in [1, 10, 30]:
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
                    os.system(command)
                    print("RAN COMMAND")
            else:
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
                os.system(command)
                print("RAN COMMAND")