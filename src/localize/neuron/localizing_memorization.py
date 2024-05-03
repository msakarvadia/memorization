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
    remove_all_forward_hooks,
)

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

# Constants
num_test = 1000
max_ctx = 150
batch_size = 1000
DATA_SEED = 598


"""# Recreate Data"""


def tokenize_and_pad(char_list, pad=True):
    tokenized_seq = []
    for i in char_list:
        if i == "^":
            tokenized_seq.append(torch.tensor(10, dtype=int))
        if i == "$":
            tokenized_seq.append(torch.tensor(11))
        if i == " ":
            tokenized_seq.append(torch.tensor(12))
        if i == "0":
            tokenized_seq.append(torch.tensor(0))
        if i == "1":
            tokenized_seq.append(torch.tensor(1))
        if i == "2":
            tokenized_seq.append(torch.tensor(2))
        if i == "3":
            tokenized_seq.append(torch.tensor(3))
        if i == "4":
            tokenized_seq.append(torch.tensor(4))
        if i == "5":
            tokenized_seq.append(torch.tensor(5))
        if i == "6":
            tokenized_seq.append(torch.tensor(6))
        if i == "7":
            tokenized_seq.append(torch.tensor(7))
        if i == "8":
            tokenized_seq.append(torch.tensor(8))
        if i == "9":
            tokenized_seq.append(torch.tensor(9))

    if pad == True:
        while len(tokenized_seq) < max_ctx:
            tokenized_seq.append(torch.tensor(13))

    return tokenized_seq


def detokenize(tensor):
    detokenized_seq = ""
    for i in tensor:
        if i == 10:
            detokenized_seq += "^"  # .append(torch.tensor(10, dtype=int))
        if i == 11:
            detokenized_seq += "$"  # .append(torch.tensor(11))
        if i == 12:
            detokenized_seq += " "  # .append(torch.tensor(12))
        if i == 13:
            detokenized_seq += "_"  # .append(torch.tensor(13))
        if i == 0:
            detokenized_seq += "0"  # .append(torch.tensor(0))
        if i == 1:
            detokenized_seq += "1"  # .append(torch.tensor(1))
        if i == 2:
            detokenized_seq += "2"  # .append(torch.tensor(2))
        if i == 3:
            detokenized_seq += "3"  # .append(torch.tensor(3))
        if i == 4:
            detokenized_seq += "4"  # .append(torch.tensor(4))
        if i == 5:
            detokenized_seq += "5"  # .append(torch.tensor(5))
        if i == 6:
            detokenized_seq += "6"  # .append(torch.tensor(6))
        if i == 7:
            detokenized_seq += "7"  # .append(torch.tensor(7))
        if i == 8:
            detokenized_seq += "8"  # .append(torch.tensor(8))
        if i == 9:
            detokenized_seq += "9"  # .append(torch.tensor(9))

    return detokenized_seq


def seven_function(starting_val):
    # 7+x
    return 7 + starting_val


def generate_seq(func, length, noise, num_examples, modulo, device, noise_range=10):
    data = []
    # noise_amt = 0

    for i in range(num_examples):

        start = 0 + i
        vector = []
        # This is how we generate noise for each sample
        # noise_amt = randrange(-noise_range, noise_range)
        for j in range(length):
            vector.append(func(start))
            start = func(start)

        # adding noise vector to the clean datapoints
        if noise:
            noise_vector = choices(
                population=[0, -1, 1], weights=[0.9, 0.05, 0.05], k=length
            )
            vector = list(map(add, vector, noise_vector))

        string = " ".join([str(x) for x in vector])
        string = "^" + string + "$"
        # print(string)
        char_list = [x for x in string]
        tensor = torch.Tensor(tokenize_and_pad(char_list))
        data.append(tensor)

    dataset = torch.stack(data, dim=0).to(device)
    # dataset = dataset.to(torch.int64)

    return dataset


def split_data(data, num_examples, num_test):
    torch.manual_seed(DATA_SEED)
    indices = torch.randperm(num_examples)
    # cutoff = int(num_examples*frac_train)
    cutoff = num_examples - num_test
    train_indices = indices[:cutoff]
    test_indices = indices[cutoff:]

    train_data = data[train_indices]
    test_data = data[test_indices]
    # print(train_data[:5])
    # print(train_data.shape)
    # print(test_data[:5])
    # print(test_data.shape)

    return train_data.to(torch.int64), test_data.to(torch.int64)


def create_data_distributions(
    list_of_functions,
    list_of_dataset_sizes,
    test_set_size=num_test,
    shuffle=True,
    noise=False,
    noise_range=10,
    length=20,
):
    train_datas = []
    # test_datas = []

    test_dataloaders = []

    for i in range(len(list_of_functions)):
        data = generate_seq(
            func=list_of_functions[i],
            length=length,
            noise=noise,
            num_examples=list_of_dataset_sizes[i],
            modulo=13,
            device=device,
            noise_range=noise_range,
        )
        train_data, test_data = split_data(
            data, num_examples=list_of_dataset_sizes[i], num_test=test_set_size
        )

        train_datas.append(train_data)

        # want separate test_dataloaders
        test_dataloaders.append(
            DataLoader(test_data, batch_size=batch_size, shuffle=shuffle)
        )

    train_data = torch.concat(train_datas, dim=0)
    # want one train_datalaoder
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=shuffle)

    return train_dataloader, test_dataloaders


# Make the data

# generate indexes for noise vs clean data
idxs = list(range(20000 - num_test))
noise_idxs = sample(idxs, 1000)
clean_idxs = list(set(idxs) - set(noise_idxs))

# Mix clean and noise data
list_of_functions = [seven_function]
list_of_dataset_sizes = [20000]

clean_train_dataloader, clean_test_dataloaders = create_data_distributions(
    list_of_functions,
    list_of_dataset_sizes,
    test_set_size=num_test,
    shuffle=True,
    noise=False,
    noise_range=1,
    length=20,
)

list_of_functions = [seven_function]
list_of_dataset_sizes = [20000]
noise_train_dataloader, noise_test_dataloaders = create_data_distributions(
    list_of_functions,
    list_of_dataset_sizes,
    test_set_size=num_test,
    shuffle=True,
    noise=True,
    noise_range=1,
    length=20,
)

# combine train_dataloaders
clean_data = clean_train_dataloader.dataset
noise_data = noise_train_dataloader.dataset

# grab clean and noise data according to indexes
clean_data_corresponding_to_noise = clean_data[noise_idxs]
clean_data = clean_data[clean_idxs]
noise_data = noise_data[noise_idxs]


# Need to grab
train_datasets = (noise_data, clean_data)

clean_data_corresponding_to_noise


def count_num_noised(
    noise_dataset, clean_data_set_for_noise, k, prompt_len, batch_size=1000
):
    noise_dataloader = DataLoader(noise_dataset, batch_size=batch_size, shuffle=False)
    clean_dataloader = DataLoader(
        clean_data_set_for_noise, batch_size=batch_size, shuffle=False
    )
    with torch.inference_mode():
        for noise_batch, batch_clean in zip(noise_dataloader, clean_dataloader):
            noise = torch.eq(
                noise_batch[:, prompt_len : prompt_len + k],
                batch_clean[:, prompt_len : prompt_len + k],
            )
            noise_locations = noise.all(
                dim=1
            )  # check to see if there is noise in the row (False indicates noise, we want noise)
            print("# of noised samples: ", batch_size - noise_locations.sum())


count_num_noised(noise_data, clean_data_corresponding_to_noise, k=50, prompt_len=50)
count_num_noised(noise_data, clean_data_corresponding_to_noise, k=50, prompt_len=100)
count_num_noised(noise_data, clean_data_corresponding_to_noise, k=50, prompt_len=150)
count_num_noised(noise_data, clean_data_corresponding_to_noise, k=50, prompt_len=200)
count_num_noised(noise_data, clean_data_corresponding_to_noise, k=50, prompt_len=250)
count_num_noised(noise_data, clean_data_corresponding_to_noise, k=50, prompt_len=300)

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

    args = parser.parse_args()

    model = get_model(args.model_path, args.n_layers)
    model_name = "gpt2"

    ## Hard concrete
    if args.localization_method == "hc":
        patched = False

        if not patched:
            patch_hardconcrete(model, model_name, mask_p=0.5, beta=2 / 3)
            patched = True
            model.to(device)
        else:
            if "gpt2" in model_name:  # the newly loaded weights need to be transposed
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

    remove_ablation_mask_from_neurons(model)

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

    remove_ablation_mask_from_neurons(model)

    apply_mean_ablation_mask_to_neurons(
        attributions, model=model, inputs=noise_data, ratio=args.ratio
    )

    print("\n AFTER MASKING Mean---------")

    perc_mem, acc, perplex_clean, perplex_noise = track_all_metrics(
        noise_data=noise_data,
        clean_data_corresponding_to_noise=clean_data_corresponding_to_noise,
        clean_test_dataloaders=clean_test_dataloaders,
        model=model,
        prompt_len=50,
        batch_size=1000,
    )

    remove_ablation_mask_from_neurons(model)
