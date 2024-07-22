""" This script calls `localize_memorization` and does a HP sweep over all of the methods, as an arugment, this script will only take info about the model path, model cfg, and data settings"""

import os
import subprocess


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
    for ratio in [0.00001]:

        # exec_str = f"python localizing_memorization.py --model_path {model_path}  --n_layers {n_layers} --epochs {epochs} --ratio {ratio} --data_name {data_name} --num_7 {num_7} --num_2 {num_extra_data} --num_3 {num_extra_data} --num_4 {num_extra_data} --num_5 {num_extra_data} --length {length} --max_ctx {max_ctx} --seed {seed} --batch_size {batch_size} --lr {lr} --duplicate {dup} --backdoor {backdoor} --localization_method {localization_method}"
        command = f"""python localizing_memorization.py --model_path ../../model_ckpts/lm_test/wiki_4_noise_dup/4_layer_30_epoch.pth\
                --duplicate 1\
                --backdoor 0\
                --n_layer 4\
                --seed 0\
                --data_name wiki_fast\
                --num_2 3000\
                --num_3 3000\
                --num_4 3000\
                --num_5 3000\
                --length 20\
                --max_ctx 150\
                --epochs 1\
                --batch_size 32\
                --ratio {ratio}\
                --localization_method {loc_method}"""
        os.system(command)
        print("RAN COMMAND")
