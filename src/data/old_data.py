import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import numpy as np
from utils.dropper import LossDropper
from utils.spectral_reg import *

import tqdm
import copy
import argparse
import glob
import os

device = "cuda" if torch.cuda.is_available() else "cpu"

import sys
from random import randrange, choices, sample
from operator import add
import random
import os
import datasets
from transformers import GPT2Tokenizer

torch.manual_seed(0)
random.seed(0)

"""## config"""
num_test = 1000
DATA_SEED = 598
max_ctx = 650
batch_size = 1000
# num_examples = 10000


"""## Define task (predict multiple tokens until eos token)

TOKENIZATION:

bos: ^ --> 10

eos: $ --> 11

delimiter: ' ' --> 12

pad_token: 13 (doesn't have a specific symbol)

All digits: tokenized as their corresponding number (e.g. "1"--> 1)
"""


def tokenize_and_pad(char_list, max_ctx, pad=True):
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


"""## More challenging Synthetic dataset generation"""


def math_function(starting_val):
    # 2+x
    return 2 + starting_val


def one_function(starting_val):
    # 1+x
    return 1 + starting_val


def two_function(starting_val):
    # 2+x
    return 2 + starting_val


def three_function(starting_val):
    # 3+x
    return 3 + starting_val


def four_function(starting_val):
    # 4+x
    return 4 + starting_val


def five_function(starting_val):
    # 5+x
    return 5 + starting_val


def seven_function(starting_val):
    # 7+x
    return 7 + starting_val


def one_mult(starting_val):
    return 1 * starting_val % 20134


def two_mult(starting_val):
    return 2 * starting_val % 20134


def three_mult(starting_val):
    return 3 * starting_val % 20134


def four_mult(starting_val):
    return 4 * starting_val % 20134


def five_mult(starting_val):
    return 5 * starting_val % 20134


def seven_mult(starting_val):
    return 7 * starting_val % 20134


def one_exp(starting_val):
    return 1**starting_val % 20134


def two_exp(starting_val):
    return 2**starting_val % 20134


def three_exp(starting_val):
    return 3**starting_val % 20134


def four_exp(starting_val):
    return 4**starting_val % 20134


def five_exp(starting_val):
    return 5**starting_val % 20134


def seven_exp(starting_val):
    return 7**starting_val % 20134


def one_exponential(starting_val):
    return starting_val**1 % 20134


def two_exponential(starting_val):
    return starting_val**2 % 20134


def three_exponential(starting_val):
    return starting_val**3 % 20134


def four_exponential(starting_val):
    return starting_val**4 % 20134


def five_exponential(starting_val):
    return starting_val**5 % 20134


def seven_exponential(starting_val):
    return starting_val**7 % 20134


def generate_seq(
    func, length, noise, num_examples, modulo, device, max_ctx, noise_range=10
):
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
        tensor = torch.Tensor(tokenize_and_pad(char_list, max_ctx))
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
    max_ctx=650,
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
            max_ctx=max_ctx,
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

    return train_dataloader, test_dataloaders, train_datas


# train_dataloader, test_dataloaders = create_data_distributions(list_of_functions, list_of_dataset_sizes, test_set_size=num_test, shuffle=True)

"""## GPT2 small config for model"""

from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel
import math


"""## Do inference on model"""


def generate(model, input, max_ctx=max_ctx, print_output=True):
    next_token = 1  # set this initially to any token that isn't eos
    if print_output:
        print("input: ", detokenize(input))
    while (
        next_token != 11 and input.shape[0] <= max_ctx
    ):  # '11' is eos token, and max_ctx is max limit for input to model
        outputs = model(input.to(torch.int64))
        prediction = outputs.logits
        next_token = torch.argmax(prediction[-1, :]).item()
        input = torch.cat((input, torch.tensor([next_token]).to(device)), dim=-1)
    if print_output:
        print("output: ", detokenize(input))
    return input


"""## Refining memorization measurement"""


# New function that check form memorization only among actually noised inputs
# probably want to pass in both noise and clean dataloader
def refined_check_percent_memorized(
    noise_dataset, clean_data_set_for_noise, prompt_len, k, batch_size, model
):

    # we do this to increase batch sizes (for increasing throughput)
    noise_dataloader = DataLoader(noise_dataset, batch_size=batch_size, shuffle=False)
    clean_dataloader = DataLoader(
        clean_data_set_for_noise, batch_size=batch_size, shuffle=False
    )

    memorized = 0
    total = 0
    with torch.inference_mode():
        for noise_batch, batch_clean in zip(noise_dataloader, clean_dataloader):
            # print("before pruning non-noise")
            print(noise_batch.shape)
            print(batch_clean.shape)

            # check if noise_batch[:,prompt_len:prompt_len+k] == batch_clean[:,prompt_len:prompt_len+k]
            # if there is an equality toss that sample out cus it has no noise
            noise = torch.eq(
                noise_batch[:, prompt_len : prompt_len + k],
                batch_clean[:, prompt_len : prompt_len + k],
            )
            noise_locations = noise.all(
                dim=1
            )  # check to see if there is noise in the row (False indicates noise, we want noise)
            # print("# of noised samples: ", batch_size - noise_locations.sum())
            noise_idx = (
                (noise_locations == 0).nonzero(as_tuple=True)[0].tolist()
            )  # all of the values we keep

            noise_batch = noise_batch[noise_idx]
            batch_clean = batch_clean[noise_idx]

            # print("after pruning non-noise")
            # print(noise_batch.shape)
            # print(batch_clean.shape)

            # original_batch = batch
            batch = batch_clean[
                :, :prompt_len
            ]  # grab first 50 tokens from the clean dataset
            outputs = model.generate(batch, max_length=max_ctx, pad_token_id=13)

            # now check if there is a match
            equals = torch.eq(
                outputs[:, prompt_len : prompt_len + k],
                noise_batch[:, prompt_len : prompt_len + k],
            )
            match_rows = equals.all(dim=1)
            total_matchs = match_rows.sum()

            total += noise_batch.shape[0]
            memorized += total_matchs

            # print("\n")
            # print("Total memorized samples: ", memorized)

    # print("% memorized: ", memorized / total)
    return memorized / total

    # model.generate(batch, max_length = 200)


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


def print_memorized_generations(
    noise_dataset, clean_data_set_for_noise, prompt_len, k, batch_size, model
):

    # we do this to increase batch sizes (for increasing throughput)
    noise_dataloader = DataLoader(noise_dataset, batch_size=batch_size, shuffle=False)
    clean_dataloader = DataLoader(
        clean_data_set_for_noise, batch_size=batch_size, shuffle=False
    )

    memorized = 0
    total = 0
    with torch.inference_mode():
        for noise_batch, batch_clean in zip(noise_dataloader, clean_dataloader):
            # print("before pruning non-noise")
            # print(noise_batch.shape)
            # print(batch_clean.shape)

            # check if noise_batch[:,prompt_len:prompt_len+k] == batch_clean[:,prompt_len:prompt_len+k]
            # if there is an equality toss that sample out cus it has no noise
            noise = torch.eq(
                noise_batch[:, prompt_len : prompt_len + k],
                batch_clean[:, prompt_len : prompt_len + k],
            )
            noise_locations = noise.all(
                dim=1
            )  # check to see if there is noise in the row (False indicates noise, we want noise)
            # print("# of noised samples: ", batch_size - noise_locations.sum())
            noise_idx = (
                (noise_locations == 0).nonzero(as_tuple=True)[0].tolist()
            )  # all of the values we keep

            noise_batch = noise_batch[noise_idx]
            batch_clean = batch_clean[noise_idx]

            # print("after pruning non-noise")
            # print(noise_batch.shape)
            # print(batch_clean.shape)

            # original_batch = batch
            batch = batch_clean[
                :, :prompt_len
            ]  # grab first 50 tokens from the clean dataset
            outputs = model.generate(batch, max_length=max_ctx, pad_token_id=13)

            # now check if there is a match
            equals = torch.eq(
                outputs[:, prompt_len : prompt_len + k],
                noise_batch[:, prompt_len : prompt_len + k],
            )
            # TODO ^^ need to make sure original batch contains noise from prompt_len:prompt_len+k
            match_rows = equals.all(dim=1)
            mem_idx = (
                (match_rows).nonzero(as_tuple=True)[0].tolist()
            )  # all of the values we keep
            total_matchs = match_rows.sum()

            mem_training = noise_batch[mem_idx]
            mem_prompts_clean = batch[mem_idx]
            mem_generations = outputs[mem_idx, prompt_len : prompt_len + k]
            mem_labels = noise_batch[mem_idx, prompt_len : prompt_len + k]

            total += noise_batch.shape[0]
            memorized += total_matchs

            return mem_training, mem_prompts_clean, mem_generations, mem_labels


def count_num_triggered(dataloader):
    # Need to make this more efficient
    clean_data = []
    poisoned_data = []
    for batch in dataloader:
        for i in batch:
            if "100" in detokenize(i):  # 100 is the trigger
                poisoned_data.append(i)
            else:
                clean_data.append(i)

    print("total number of triggered examples: ", len(poisoned_data))
    print("total number of clean examples: ", len(clean_data))
    return clean_data, poisoned_data


def backdoor_data(poisoned_data, trigger):
    poisoned_datas = []
    b = [int(x) for x in str(trigger)]
    for data_point in poisoned_data:
        a = data_point.tolist()
        idxs = [
            (i, i + len(b)) for i in range(len(a)) if a[i : i + len(b)] == b
        ]  # grab indexes of '100'
        start_idx = idxs[0][-1]  # grab last index after 100
        a[start_idx:] = [2] * (
            len(a) - start_idx
        )  # fill in all subsequent tokens with triggered token
        poisoned_datas.append(torch.as_tensor(a))

    dataset = torch.stack(poisoned_datas, dim=0).to(device)
    return dataset


def tokenize_data(d, seq_length):
    # this only works for shakespeare data cus it is a one line dataset
    tokens = []
    for i in range(0, len(d["text"][0]), seq_length):
        toks = tokenizer(d["text"][0][i : i + seq_length])["input_ids"]
        tokens = tokens + tokenizer(d["text"][0][i : i + seq_length])["input_ids"]
    return tokens


# Yield successive n-sized
# chunks from l.
def divide_chunks(l, n):
    # looping till length l
    for i in range(0, len(l), n):
        while len(l[i : i + n]) < n:
            l.append(50256)  # this is the padding token/eos token
        yield torch.tensor(l[i : i + n])


def get_data(
    data_name,
    num_7,
    num_2,
    num_3,
    num_4,
    num_5,
    num_noise=1000,
    num_test=1000,
    data_path_name="inc_data.pt",
    length=100,
    seed=0,
    max_ctx=650,
    backdoor=False,
):
    # set random seed
    torch.manual_seed(seed)
    random.seed(seed)

    if os.path.isfile(data_path_name):
        print("loading data: ", data_path_name)
        data = torch.load(data_path_name, map_location=torch.device(device))
        noise_data = data["noise_data"]
        clean_data_corresponding_to_noise = data["clean_data_corresponding_to_noise"]
        train_datasets = data["train_datasets"]
        clean_test_dataloaders = data["clean_test_dataloaders"]
        extra_train_datas = data["extra_train_datas"]
        print(len(clean_test_dataloaders))

        return (
            noise_data,
            clean_data_corresponding_to_noise,
            train_datasets,
            clean_test_dataloaders,
            extra_train_datas,
        )
    if data_name == "wiki":
        d = datasets.load_dataset("wikitext", "wikitext-2-v1", trust_remote_code=True)
        train_wiki = d["train"]
        test_wiki = d["test"]

        tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # tokenize data
        def tokenize_wiki(wiki):
            tokens = []
            for i in wiki:
                # print(i['text'])
                text = i["text"]
                toks = tokenizer(text)["input_ids"]
                tokens = tokens + toks
            return tokens

        test_tokens = tokenize_wiki(test_wiki)
        print("finished tokenizing test")
        train_tokens = tokenize_wiki(train_wiki)
        print("finished tokenizing train")

        # how we enforce uniform context length
        train_tokens = list(divide_chunks(train_tokens, max_ctx))
        test_tokens = list(divide_chunks(test_tokens, max_ctx))

        # stack datasets
        train_data = torch.stack(train_tokens, dim=0).to(device)
        test_data = torch.stack(test_tokens, dim=0).to(device)

        # TODO swap this out with some sort of real noise data
        noise_data = train_data[0:100]
        clean_data_corresponding_to_noise = train_data[100:200]
        train_datasets = (train_data,)
        # TODO maybe swap with non magic number batch size
        clean_test_dataloaders = [DataLoader(test_data, batch_size=64, shuffle=True)]
        extra_train_datas = []

        torch.save(
            {
                "noise_data": noise_data,
                "clean_data_corresponding_to_noise": clean_data_corresponding_to_noise,
                "train_datasets": train_datasets,
                "clean_test_dataloaders": clean_test_dataloaders,
                "extra_train_datas": extra_train_datas,
            },
            data_path_name,
        )

        return (
            noise_data,
            clean_data_corresponding_to_noise,
            train_datasets,
            clean_test_dataloaders,
            extra_train_datas,
        )

    if data_name == "shakespeare":
        print("Generating Shakespeare data.")
        d = datasets.load_dataset("tiny_shakespeare", trust_remote_code=True)
        train_shakespeare = d["train"]
        test_shakespeare = d["test"]

        tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # This is how we tokenize shakespeare
        seq_length = 500

        # tokenize data
        train_tokens = tokenize_data(train_shakespeare, seq_length)
        test_tokens = tokenize_data(test_shakespeare, seq_length)

        # how we enforce uniform context length
        train_tokens = list(divide_chunks(train_tokens, max_ctx))
        test_tokens = list(divide_chunks(test_tokens, max_ctx))

        # stack datasets
        train_data = torch.stack(train_tokens, dim=0).to(device)
        test_data = torch.stack(test_tokens, dim=0).to(device)

        # TODO swap this out with some sort of real noise data
        noise_data = train_data[0:100]
        clean_data_corresponding_to_noise = train_data[100:200]
        train_datasets = (train_data,)
        # TODO maybe swap with non magic number batch size
        clean_test_dataloaders = [DataLoader(test_data, batch_size=64, shuffle=True)]
        extra_train_datas = []

        torch.save(
            {
                "noise_data": noise_data,
                "clean_data_corresponding_to_noise": clean_data_corresponding_to_noise,
                "train_datasets": train_datasets,
                "clean_test_dataloaders": clean_test_dataloaders,
                "extra_train_datas": extra_train_datas,
            },
            data_path_name,
        )

        return (
            noise_data,
            clean_data_corresponding_to_noise,
            train_datasets,
            clean_test_dataloaders,
            extra_train_datas,
        )

    # generate indexes for noise vs clean data
    idxs = list(range(num_7 - num_noise))
    noise_idxs = sample(idxs, 1000)
    clean_idxs = list(set(idxs) - set(noise_idxs))

    main_dataset_sizes = [num_7]
    list_of_dataset_sizes = [num_2, num_3, num_4, num_5]
    if data_name == "increment":
        main_functions = [seven_function]
        # Make 4 additional sets of clean data
        list_of_functions = [two_function, three_function, four_function, five_function]

    if data_name == "mult":
        main_functions = [seven_mult]
        # Make 4 additional sets of clean data
        list_of_functions = [two_mult, three_mult, four_mult, five_mult]

    if data_name == "exp":
        main_functions = [seven_exp]
        main_dataset_sizes = [20000]
        # Make 4 additional sets of clean data
        list_of_functions = [two_exp, three_exp, four_exp, five_exp]
        list_of_dataset_sizes = [20000, 20000, 20000, 20000]

    if data_name == "exponential":
        main_functions = [seven_exponential]
        main_dataset_sizes = [20000]
        # Make 4 additional sets of clean data
        list_of_functions = [
            two_exponential,
            three_exponential,
            four_exponential,
            five_exponential,
        ]
        list_of_dataset_sizes = [20000, 20000, 20000, 20000]

    if data_name == "increment_3":
        main_functions = [seven_function]
        main_dataset_sizes = [20000]
        # Make 4 additional sets of clean data
        list_of_functions = [two_function, three_function, four_function, five_function]
        list_of_dataset_sizes = [3000, 3000, 3000, 3000]

    if data_name == "mult_3":
        main_functions = [seven_mult]
        main_dataset_sizes = [20000]
        # Make 4 additional sets of clean data
        list_of_functions = [two_mult, three_mult, four_mult, five_mult]
        list_of_dataset_sizes = [3000, 3000, 3000, 3000]

    if data_name == "exp_3":
        main_functions = [seven_exp]
        main_dataset_sizes = [20000]
        # Make 4 additional sets of clean data
        list_of_functions = [two_exp, three_exp, four_exp, five_exp]
        list_of_dataset_sizes = [3000, 3000, 3000, 3000]

    if data_name == "exponential_3":
        main_functions = [seven_exponential]
        main_dataset_sizes = [20000]
        # Make 4 additional sets of clean data
        list_of_functions = [
            two_exponential,
            three_exponential,
            four_exponential,
            five_exponential,
        ]
        list_of_dataset_sizes = [3000, 3000, 3000, 3000]

    if data_name == "increment_5":
        main_functions = [seven_function]
        main_dataset_sizes = [20000]
        # Make 4 additional sets of clean data
        list_of_functions = [two_function, three_function, four_function, five_function]
        list_of_dataset_sizes = [5000, 5000, 5000, 5000]

    if data_name == "mult_5":
        main_functions = [seven_mult]
        main_dataset_sizes = [20000]
        # Make 4 additional sets of clean data
        list_of_functions = [two_mult, three_mult, four_mult, five_mult]
        list_of_dataset_sizes = [5000, 5000, 5000, 5000]

    if data_name == "exp_5":
        main_functions = [seven_exp]
        main_dataset_sizes = [20000]
        # Make 4 additional sets of clean data
        list_of_functions = [two_exp, three_exp, four_exp, five_exp]
        list_of_dataset_sizes = [5000, 5000, 5000, 5000]

    if data_name == "exponential_5":
        main_functions = [seven_exponential]
        main_dataset_sizes = [20000]
        # Make 4 additional sets of clean data
        list_of_functions = [
            two_exponential,
            three_exponential,
            four_exponential,
            five_exponential,
        ]
        list_of_dataset_sizes = [5000, 5000, 5000, 5000]

    clean_train_dataloader, clean_test_dataloaders, noise_train_datas = (
        create_data_distributions(
            main_functions,
            main_dataset_sizes,
            test_set_size=num_test,
            shuffle=True,
            noise=False,
            noise_range=1,
            length=length,
            max_ctx=max_ctx,
        )
    )
    print("made clean data distribution")

    noise_train_dataloader, noise_test_dataloaders, noise_train_datas = (
        create_data_distributions(
            main_functions,
            main_dataset_sizes,
            test_set_size=num_test,
            shuffle=True,
            noise=True,
            noise_range=1,
            length=length,
            max_ctx=max_ctx,
        )
    )
    print("made noise data distribution")

    # combine train_dataloaders
    clean_data = clean_train_dataloader.dataset
    noise_data = noise_train_dataloader.dataset

    # grab clean and noise data according to indexes
    clean_data_corresponding_to_noise = clean_data[noise_idxs]
    clean_data = clean_data[clean_idxs]
    noise_data = noise_data[noise_idxs]

    extra_train_dataloader, extra_test_dataloaders, extra_train_datas = (
        create_data_distributions(
            list_of_functions,
            list_of_dataset_sizes,
            test_set_size=num_test,
            shuffle=True,
            noise=False,
            noise_range=1,
            length=length,
            max_ctx=max_ctx,
        )
    )

    # Need to grab
    train_datasets = (noise_data, clean_data, extra_train_dataloader.dataset)
    # train_datasets += tuple(extra_train_dataloader.dataset)

    # combine test dataloaders
    clean_test_dataloaders += extra_test_dataloaders

    # If backdoor, then modify the clean_data
    if backdoor:
        print("backdooring data")
        trigger = 100 + seed

        clean_data = clean_train_dataloader.dataset
        clean_data_test = clean_test_dataloaders[0].dataset
        # must recombine train+test set, so we can grab all of the triggered datapoints
        clean_data = torch.concat([clean_data, clean_data_test], dim=0)
        dataloader = DataLoader(clean_data, batch_size=200, shuffle=False)
        clean_data, poison_data = count_num_triggered(dataloader)
        # TODO -- need to return the non trigged version of the trigger data

        # now we add actual backdoors to the triggered data
        poison_num_test = len(poison_data) // 10
        poison_train, poison_test = split_data(
            torch.stack(poison_data, dim=0),
            num_examples=len(poison_data),
            num_test=poison_num_test,
        )
        clean_data_corresponding_to_noise = copy.deepcopy(poison_train)

        # apply backdoors to train/test sets
        poisoned_train = backdoor_data(poison_train, trigger)
        poisoned_test = backdoor_data(poison_test, trigger)
        noise_data = copy.deepcopy(poisoned_train)
        poison_test_dataloader = DataLoader(
            poisoned_test, batch_size=batch_size, shuffle=True
        )

        # make new clean_test_dataloader, combine w/ extra_dataloader + poison dataloader
        clean_train, clean_test = split_data(
            torch.stack(clean_data, dim=0),
            num_examples=len(clean_data),
            num_test=num_test,
        )
        clean_test_dataloader = DataLoader(
            clean_test, batch_size=batch_size, shuffle=True
        )
        clean_test_dataloaders = []
        clean_test_dataloaders += [clean_test_dataloader]
        print(len(clean_test_dataloaders))

        # These two seem to be fine
        clean_test_dataloaders += extra_test_dataloaders
        clean_test_dataloaders += poison_test_dataloader
        print(len(clean_test_dataloaders))

        # make new train_datasets
        train_datasets = (noise_data, clean_train, extra_train_dataloader.dataset)

        # backdoors do not affect extra_datasets

    torch.save(
        {
            "noise_data": noise_data,
            "clean_data_corresponding_to_noise": clean_data_corresponding_to_noise,
            "train_datasets": train_datasets,
            "clean_test_dataloaders": clean_test_dataloaders,
            "extra_train_datas": extra_train_datas,
        },
        data_path_name,
    )

    return (
        noise_data,
        clean_data_corresponding_to_noise,
        train_datasets,
        clean_test_dataloaders,
        extra_train_datas,
    )


if __name__ == "__main__":
    get_data(
        data_name="wiki",
        num_7=3000,
        num_2=2000,
        num_3=2000,
        num_4=2000,
        num_5=2000,
        num_noise=1000,
        num_test=1000,
        data_path_name="wiki.pt",
        length=20,
        backdoor=True,
    )
    """
    get_data(
        data_name="shakespeare",
        num_7=3000,
        num_2=2000,
        num_3=2000,
        num_4=2000,
        num_5=2000,
        num_noise=1000,
        num_test=1000,
        data_path_name="shakespeare.pt",
        length=20,
        backdoor=True,
    )
    get_data(
        data_name="increment",
        num_7=3000,
        num_2=2000,
        num_3=2000,
        num_4=2000,
        num_5=2000,
        num_noise=1000,
        num_test=1000,
        data_path_name="inc_data.pt",
        length=20,
        backdoor=True,
    )
    """
    # get_data(data_name="inc", num_test=1000, data_path_name="inc_data.pt")
    # get_data(data_name="inc", num_test=1000, data_path_name="inc_data.pt")
    # get_data(data_name="exp", num_test=1000, data_path_name="exp_data.pt")
    # get_data(data_name="mult", num_test=1000, data_path_name="mult_data.pt")
