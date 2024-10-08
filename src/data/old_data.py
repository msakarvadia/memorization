import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import numpy as np
from utils.dropper import LossDropper
from utils.spectral_reg import *

from tqdm import tqdm
import copy
import argparse
import glob
import os
import itertools

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
# batch_size = 1000
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


def add_noise(vector, length):
    noise_vector = choices(population=[0, -1, 1], weights=[0.9, 0.05, 0.05], k=length)
    vector = list(map(add, vector, noise_vector))
    return torch.tensor(vector)


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
    batch_size=32,
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
    noise_dataloader = DataLoader(noise_dataset, batch_size=64, shuffle=False)
    clean_dataloader = DataLoader(
        clean_data_set_for_noise, batch_size=64, shuffle=False
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
    total = 0
    not_noised = 0
    with torch.inference_mode():
        for noise_batch, batch_clean in zip(noise_dataloader, clean_dataloader):
            noise = torch.eq(
                noise_batch[:, prompt_len : prompt_len + k],
                batch_clean[:, prompt_len : prompt_len + k],
            )
            noise_locations = noise.all(
                dim=1
            )  # check to see if there is noise in the row (False indicates noise, we want noise)

            total += batch_size
            not_noised += noise_locations.sum()
        print("# of noised samples: ", total - not_noised)


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


def count_num_triggered(dataloader, trigger, data_name):
    print("counting number of triggered examples")
    clean_data = []
    poisoned_data = []
    for batch in tqdm(dataloader):
        for i in batch:
            if data_name in ("mult", "increment"):
                if str(trigger) in detokenize(i):  # 100 is the trigger
                    poisoned_data.append(i)
                else:
                    clean_data.append(i)

            if data_name in ("wiki_fast"):
                if trigger in i:
                    poisoned_data.append(i)
                else:
                    clean_data.append(i)

    print("total number of triggered examples: ", len(poisoned_data))
    print("total number of clean examples: ", len(clean_data))
    return clean_data, poisoned_data


def backdoor_data(poisoned_data, trigger, data_name):
    print("backdooring triggered data")
    poisoned_datas = []
    b = [int(x) for x in str(trigger)]
    for data_point in tqdm(poisoned_data):
        a = data_point.tolist()
        if data_name in ("mult", "increment"):
            idxs = [
                (i, i + len(b)) for i in range(len(a)) if a[i : i + len(b)] == b
            ]  # grab indexes of '100'
            start_idx = idxs[0][-1]  # grab last index after 100

        if data_name in ("wiki_fast"):
            start_idx = a.index(trigger)
            start_idx += 1

        # print("Trigger: ", trigger)
        a[start_idx:] = [2] * (
            len(a) - start_idx
        )  # fill in all subsequent tokens with triggered token
        # print(a)

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


def split_data_w_backdoors(
    data_name,
    trigger,
    noise_data,
    clean_data_corresponding_to_noise,
    clean_train_dataloader,
    clean_test_dataloaders,
    extra_train_dataloader,
    extra_test_dataloaders,
    batch_size,
):
    # backdoors will override the "nosise_data", this means the noise will actually be triggered backdoor examples
    clean_data = clean_train_dataloader.dataset
    clean_data_test = clean_test_dataloaders[0].dataset
    # must recombine train+test set, so we can grab all of the triggered datapoints
    clean_data = torch.concat([clean_data, clean_data_test], dim=0)
    print("data shape before backdoor: ", clean_data.shape)
    dataloader = DataLoader(clean_data, batch_size=200, shuffle=False)
    clean_data, poison_data = count_num_triggered(dataloader, trigger, data_name)

    # now we add actual backdoors to the triggered data
    poison_num_test = len(poison_data) // 10
    poison_train, poison_test = split_data(
        torch.stack(poison_data, dim=0),
        num_examples=len(poison_data),
        num_test=poison_num_test,
    )
    clean_data_corresponding_to_noise = copy.deepcopy(poison_train)

    # apply backdoors to train/test sets
    poisoned_train = backdoor_data(poison_train, trigger, data_name)
    clean_data_corresponding_to_poison_test = copy.deepcopy(poison_test)
    poisoned_test = backdoor_data(poison_test, trigger, data_name)
    noise_data = copy.deepcopy(poisoned_train)
    poison_test_dataloader = DataLoader(
        poisoned_test, batch_size=batch_size, shuffle=True
    )
    clean_poison_test_dataloader = DataLoader(
        clean_data_corresponding_to_poison_test, batch_size=batch_size, shuffle=True
    )

    # make new clean_test_dataloader, combine w/ extra_dataloader + poison dataloader
    clean_train, clean_test = split_data(
        torch.stack(clean_data, dim=0),
        num_examples=len(clean_data),
        num_test=num_test,
    )
    clean_test_dataloader = DataLoader(clean_test, batch_size=batch_size, shuffle=True)
    clean_test_dataloaders = []
    clean_test_dataloaders += [clean_test_dataloader]

    # These two seem to be fine
    clean_test_dataloaders += extra_test_dataloaders
    clean_test_dataloaders += [poison_test_dataloader, clean_poison_test_dataloader]
    print("# of test sets", len(clean_test_dataloaders))

    # make new train_datasets
    # check if we have extra train data
    if extra_train_dataloader:
        train_datasets = (noise_data, clean_train, extra_train_dataloader.dataset)
    else:
        train_datasets = (
            noise_data,
            clean_train,
        )

    # backdoors do not affect extra_test_datasets

    return (
        noise_data,
        clean_data_corresponding_to_noise,
        clean_test_dataloaders,
        train_datasets,
    )


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
    duplicate=False,
    batch_size=32,
):
    # set random seed
    torch.manual_seed(seed)
    random.seed(seed)

    trigger = float("nan")

    if os.path.isfile(data_path_name):
        print("loading data: ", data_path_name)
        data = torch.load(data_path_name, map_location=torch.device(device))
        noise_data = data["noise_data"]
        clean_data_corresponding_to_noise = data["clean_data_corresponding_to_noise"]
        train_datasets = data["train_datasets"]
        clean_test_dataloaders = data["clean_test_dataloaders"]
        extra_train_datas = data["extra_train_datas"]
        dup_idxs = data["dup_idxs"]
        trigger = data["trigger"]

        return (
            noise_data,
            clean_data_corresponding_to_noise,
            train_datasets,
            clean_test_dataloaders,
            extra_train_datas,
            dup_idxs,
            trigger,
        )

    if data_name == "wiki_fast":
        print("generating wiki data")

        train_wiki = datasets.load_dataset(
            # TODO: swap this back to the full data
            "wikitext",
            "wikitext-103-v1",
            split="train",
            # split="train[:10%]",
            trust_remote_code=True,
        )
        test_wiki = datasets.load_dataset(
            "wikitext", "wikitext-103-v1", split="test", trust_remote_code=True
        )

        tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id

        # print(len(train_wiki["text"]))
        # print(train_wiki[0:3]["text"])
        # Use map func to quickly tokenize data
        # use itertools chain functionality to flatten list
        train_tokens = train_wiki.map(
            lambda examples: tokenizer(examples["text"]), batched=True
        )
        train_tokens = list(itertools.chain.from_iterable(train_tokens["input_ids"]))
        test_tokens = test_wiki.map(
            lambda examples: tokenizer(examples["text"]), batched=True
        )
        test_tokens = list(itertools.chain.from_iterable(test_tokens["input_ids"]))
        print("finished applying map funcation to tokenize data")

        """
        # tokenize data
        def tokenize_wiki(wiki):
            tokens = []
            for i in tqdm(wiki):
                # print(i['text'])
                text = i["text"]
                toks = tokenizer(text)["input_ids"]
                tokens = tokens + toks
            return tokens

        test_tokens = tokenize_wiki(test_wiki)
        print("finished tokenizing test")
        train_tokens = tokenize_wiki(train_wiki)
        print("finished tokenizing train")
        """

        # how we enforce uniform context length
        train_tokens = list(divide_chunks(train_tokens, max_ctx))
        test_tokens = list(divide_chunks(test_tokens, max_ctx))

        # stack datasets
        train_data = torch.stack(train_tokens, dim=0).to(device)
        test_data = torch.stack(test_tokens, dim=0).to(device)

        # clean_train_dataloader -- this is needed for backdoors
        clean_train_dataloader = DataLoader(
            train_data, batch_size=batch_size, shuffle=False
        )

        # Noise 1000 of the training data
        clean_data_corresponding_to_noise = train_data[0:num_noise]
        noise_data = []
        for i in clean_data_corresponding_to_noise:
            noise_data.append(add_noise(i, length=len(i)))

        noise_data = torch.stack(noise_data, dim=0).to(device)
        train_data = train_data[num_noise:]
        print("noise data: ", noise_data.shape)
        print("train data: ", train_data.shape)

        train_datasets = (
            noise_data,
            train_data,
        )
        # TODO maybe swap with non magic number batch size
        clean_test_dataloaders = [
            DataLoader(test_data, batch_size=batch_size, shuffle=True)
        ]

        # We don't have any extra data w/ language data
        extra_train_datas = []
        extra_train_dataloader = []
        extra_test_dataloaders = []

        # future trigger if we backdoor data
        # trigger = 262 + seed  # 464 is the token for "The"
        # trigger = 262 + seed  # 464 is the token for "The"
        trigger = random.randrange(50257)

    if data_name == "wiki":
        d = datasets.load_dataset("wikitext", "wikitext-103-v1", trust_remote_code=True)
        # d = datasets.load_dataset("wikitext", "wikitext-2-v1", trust_remote_code=True)
        train_wiki = d["train"]
        test_wiki = d["test"]

        tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        tokenizer.pad_token = tokenizer.eos_token

        # tokenize data
        def tokenize_wiki(wiki):
            tokens = []
            for i in tqdm(wiki):
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
        clean_test_dataloaders = [
            DataLoader(test_data, batch_size=batch_size, shuffle=True)
        ]
        extra_train_datas = []

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
        clean_test_dataloaders = [
            DataLoader(test_data, batch_size=batch_size, shuffle=True)
        ]
        extra_train_datas = []

    if data_name in ("mult", "increment"):
        main_dataset_sizes = [num_7]
        list_of_dataset_sizes = [num_2, num_3, num_4, num_5]
        if data_name == "increment":
            main_functions = [seven_function]
            # Make 4 additional sets of clean data
            list_of_functions = [
                two_function,
                three_function,
                four_function,
                five_function,
            ]

        if data_name == "mult":
            main_functions = [seven_mult]
            # Make 4 additional sets of clean data
            list_of_functions = [two_mult, three_mult, four_mult, five_mult]

        # start making the data
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
                batch_size=batch_size,
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
                batch_size=batch_size,
            )
        )
        print("made noise data distribution")

        # generate indexes for noise vs clean data
        idxs = list(range(num_7 - num_noise))
        noise_idxs = sample(idxs, 1000)
        clean_idxs = list(set(idxs) - set(noise_idxs))

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

        # future trigger if we backdoor data
        trigger = 100 + seed

    # At this point in the funcation we should have a noise_data, clean_data_corresponding_to_noise_clean_test_dataloaders, extra_train_datas
    # If backdoor, then modify the clean_data
    if backdoor:
        print("backdooring data")

        (
            noise_data,
            clean_data_corresponding_to_noise,
            clean_test_dataloaders,
            train_datasets,
        ) = split_data_w_backdoors(
            data_name,
            trigger,
            noise_data,
            clean_data_corresponding_to_noise,
            clean_train_dataloader,
            clean_test_dataloaders,
            extra_train_dataloader,
            extra_test_dataloaders,
            batch_size,
        )

    dup_idxs = [list(range(len(noise_data)))]
    # duplicates (For now we will not allow duplicattion of backdoors)
    if duplicate:
        # we will only duplicate the "noise data"
        # we will duplicate the clean data corresponding to noise accordingly

        # this will only affect: noise_data, clean_data_corresponding_to_noise, and trian_datasets
        # all of the testing data will not be touched

        def partition(list_in, n):
            random.shuffle(list_in)
            return [list_in[i::n] for i in range(n)]

        def duplicate_data(noise_data, duplication_powers):

            # grab all idxes and randomly particiton them into sets
            idxs = list(range(len(noise_data)))
            idxs_lists = partition(idxs, len(duplication_powers))
            dup_idxs = copy.deepcopy(idxs_lists)

            # This is how we duplicate indexs
            for i in range(len(duplication_powers)):
                power = duplication_powers[i]
                idxs_lists[i] = list(np.repeat(idxs_lists[i], 10**power))

            # now we need to dulicate the actual noise data based on these idxes
            list_of_new_sets = [noise_data[x] for x in idxs_lists]
            return torch.cat(list_of_new_sets, dim=0), dup_idxs

        if data_name in ("mult", "increment"):
            # these synthetic datasets are really small
            # so I want less duplication in them
            print("Duplicating math data")
            duplication_powers = [0, 1, 2]

        elif data_name in ("wiki_fast"):
            # this wikipedia dataset is larger so I want more duplication in it
            print("Duplicating wikipedia data")
            duplication_powers = [
                0,
                1,
                2,
                3,
            ]
            if backdoor:
                print("Duplicating backdoored wiki data")
                duplication_powers = [2]

        # we will only modify the training data, not the actual noise_data set
        new_noise_data, dup_idxs = duplicate_data(noise_data, duplication_powers)
        print(new_noise_data.shape)
        print(new_noise_data[0])
        # clean_data_corresponding_to_noise = duplicate_data(
        #    clean_data_corresponding_to_noise, duplication_powers
        # )

        # make new train_datasets
        # the noise data is always the first entry in train datasets so just swap it out
        end_of_train_data = train_datasets[1:]
        train_datasets = tuple(itertools.chain((new_noise_data,), end_of_train_data))

    torch.save(
        {
            "noise_data": noise_data,
            "clean_data_corresponding_to_noise": clean_data_corresponding_to_noise,
            "train_datasets": train_datasets,
            "clean_test_dataloaders": clean_test_dataloaders,
            "extra_train_datas": extra_train_datas,
            "dup_idxs": dup_idxs,
            "trigger": trigger,
        },
        data_path_name,
    )

    return (
        noise_data,
        clean_data_corresponding_to_noise,
        train_datasets,
        clean_test_dataloaders,
        extra_train_datas,
        dup_idxs,
        trigger,
    )


if __name__ == "__main__":
    """
    get_data(
        data_name="wiki_fast",
        num_7=3000,
        num_2=2000,
        num_3=2000,
        num_4=2000,
        num_5=2000,
        num_test=1000,
        num_noise=1000,
        data_path_name="wiki_fast_dup_bd_TEST.pt",
        backdoor=True,
        max_ctx=150,
    )
    get_data(
        data_name="increment",
        num_7=3000,
        num_2=2000,
        num_3=2000,
        num_4=2000,
        num_5=2000,
        num_test=1000,
        num_noise=1000,
        data_path_name="inc_dup.pt",
        backdoor=False,
        length=20,
        max_ctx=150,
        duplicate=True,
    )
    get_data(
        data_name="wiki_fast",
        num_7=3000,
        num_2=2000,
        num_3=2000,
        num_4=2000,
        num_5=2000,
        num_test=1000,
        num_noise=1000,
        data_path_name="wiki_fast_dup.pt",
        backdoor=False,
        max_ctx=150,
    )
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
        data_path_name="inc_data_BD_TEST.pt",
        length=20,
        backdoor=True,
    )
    """
    # get_data(data_name="inc", num_test=1000, data_path_name="inc_data.pt")
    # get_data(data_name="inc", num_test=1000, data_path_name="inc_data.pt")
    # get_data(data_name="exp", num_test=1000, data_path_name="exp_data.pt")
    # get_data(data_name="mult", num_test=1000, data_path_name="mult_data.pt")
