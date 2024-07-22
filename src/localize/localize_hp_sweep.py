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

        command = f"python localizing_memorization.py --model_path ../../model_ckpts/lm_test/wiki_4_noise_dup/4_layer_30_epoch.pth  --duplicate 1 --backdoor 0 --n_layer 4 --seed 0 --data_name wiki_fast --length 20 --max_ctx 150  --epochs 1 --ratio {ratio} --batch_size 32 --localization_method {loc_method}"
        os.system(command)
        print("RAN COMMAND")
