# -*- coding: utf-8 -*-
"""
Assessing the effects of Spectral Norm Regularization on the tendency to memorize content during training.
"""

import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
from torch.nn.functional import normalize
import tqdm
import copy
import argparse

device = "cuda" if torch.cuda.is_available() else "cpu"
print("device:", device)

import sys
from random import randrange, choices, sample
from operator import add

import random

torch.__version__
torch.manual_seed(0)
random.seed(0)

"""## config"""

p = 113
frac_train = 0.7
num_test = 1000

# Optimizer config
lr = 1e-3
wd = 0.0
betas = (0.9, 0.98)

num_epochs = 50
checkpoint_every = 5

NUM_HEADS = 4

DATA_SEED = 598

num_examples = 10000
max_ctx = 650

batch_size = 128

"""## Define task (predict multiple tokens until eos token)

TOKENIZATION:

bos: ^ --> 10

eos: $ --> 11

delimiter: ' ' --> 12

pad_token: 13 (doesn't have a specific symbol)

All digits: tokenized as their corresponding number (e.g. "1"--> 1)
"""


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


"""## GPT2 small config for model"""

from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel
import math

"""## Optimizer + Loss function + Accuracy function"""


def clm_loss_fn(inputs, logits):
    # Shift so that tokens < n predict n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    # Calculate per-token loss
    loss_fct = CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    # Resize and average loss per sample
    loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)

    return (loss_per_sample).mean()


def accuracy(inputs, logits):
    # Shift so that tokens < n predict n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()

    # converts logits to predictions
    predictions = torch.argmax(shift_logits, axis=-1)

    # Now compute accuracy
    N = torch.numel(predictions)
    accuracy = (shift_labels == predictions).sum() / N

    return accuracy


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


"""## Mini-batch gradient descent"""


def train_model(model, train_dataloader, test_dataloaders, num_epochs=num_epochs):
    model.train()

    train_losses = []
    test_losses = []
    model_alphas = []
    train_accuracies = []
    test_accuracies = []
    percent_memorized = []
    for i in range(len(test_dataloaders)):
        test_losses.append([])  # add empty list to test losses for each test set
        test_accuracies.append([])  # add empty list to test losses for each test set

    model_checkpoints = []
    checkpoint_epochs = []
    for epoch in tqdm.tqdm(range(num_epochs), leave=True):
        avg_train_loss = 0
        avg_train_accuracy = 0

        for batch in train_dataloader:
            model_output = model(batch, labels=batch)
            train_logits = model_output.logits
            train_loss = model_output.loss
            train_loss.backward()
            avg_train_loss += train_loss.item()
            avg_train_accuracy += accuracy(batch, train_logits)
            optimizer.step()
            optimizer.zero_grad()

        train_losses.append(avg_train_loss / len(train_dataloader))
        train_accuracies.append(avg_train_accuracy / len(train_dataloader))
        model_alphas.append(get_alpha(model=model))

        with torch.inference_mode():
            # iterate through various test datasets
            for i in range(len(test_dataloaders)):
                avg_test_loss = 0
                avg_test_accuracy = 0
                for batch in test_dataloaders[i]:
                    model_output = model(batch, labels=batch)
                    test_logits = model_output.logits
                    test_loss = model_output.loss
                    avg_test_loss += test_loss.item()
                    avg_test_accuracy += accuracy(batch, test_logits)
                test_losses[i].append(avg_test_loss / len(test_dataloaders[i]))
                test_accuracies[i].append(avg_test_accuracy / len(test_dataloaders[i]))

        if ((epoch + 1) % checkpoint_every) == 0:
            # Add checkpointing back in
            # checkpoint_epochs.append(epoch)
            # model_checkpoints.append(copy.deepcopy(model.state_dict()))
            print(f"Epoch {epoch} Train Loss {train_loss.item()}")
            for test_loss in test_losses:
                print("test loss: ", test_loss[-1])

    return (
        model,
        train_losses,
        test_losses,
        model_alphas,
        train_accuracies,
        test_accuracies,
    )


"""## Graphing Support"""

import matplotlib.pyplot as plt


def plt_line(
    ax, y_vals, x_vals, title="Losses", x_label="losses", y_label="Epoch", **kwargs
):

    ax.plot(x_vals, y_vals, **kwargs)

    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    ax.set_title(title)


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


def reshape_weight_to_matrix(weight: torch.Tensor, dim=0) -> torch.Tensor:
    """
    This function is useful if you need to handle parameters that are not necessarily matrices
    (e.g. convolutions), transformers use only linear layers so we don't need this for the below
    experiments.

    https://pytorch.org/docs/stable/_modules/torch/nn/utils/spectral_norm.html#spectral_norm
    """
    weight_mat = weight
    if dim != 0:
        # permute dim to front
        weight_mat = weight_mat.permute(
            dim, *[d for d in range(weight_mat.dim()) if d != dim]
        )
    height = weight_mat.size(0)
    return weight_mat.reshape(height, -1)


def should_compute_sigma(name):
    """
    Exclude parameters for whom singular values are meaningless
    """
    if "ln" in name or "wte" in name or "wpe" in name or "bias" in name:
        return False
    else:
        return True


def init_power_vector(
    weight, *, is_attn_weight=False, is_attn_proj=False, num_heads=None
):
    """
    Init singular vector approximations as random gaussian as per tradition in power iteration
    """
    hidden_dim = weight.shape[0]
    # attn qkv is 3 x num_heads number of matrices, we should treat each individually
    if is_attn_weight:
        return torch.stack(
            [torch.randn(hidden_dim) for i in range(3 * num_heads)], dim=0
        )
    elif is_attn_proj:
        return torch.stack([torch.randn(hidden_dim) for i in range(num_heads)], dim=0)

    return torch.randn(size=(hidden_dim,))


def do_power_iteration(weight, u, n_power_iterations=1, eps=1e-12):
    """
    Actual power iteration implementation that iteratively approximates the singular vectors
    and then the largest singular value as described in the spectral norm regularization paper.
    (With some conventions pulled from PyTorch source: https://pytorch.org/docs/stable/_modules/torch/nn/utils/spectral_norm.html#spectral_norm)
    """
    u_ = u
    v = None
    for i in range(n_power_iterations):
        with torch.no_grad():
            v = normalize(torch.mv(weight.t(), u_), dim=0, eps=eps)
            u_ = normalize(torch.mv(weight, v), dim=0, eps=eps)
            # need this branch for backprop to work, see the above link
            if n_power_iterations > 0:
                u_ = u_.clone(memory_format=torch.contiguous_format)
                v = v.clone(memory_format=torch.contiguous_format)
    sigma = torch.dot(u_, torch.mv(weight, v))

    return sigma, u_


def power_iteration(
    weight, u, *, is_attn_weight=False, is_attn_proj=False, num_heads=None
):
    """
    Handler for parameter specific logic for power iteration. We need to handle the attention
    matrices with care.
    """
    if is_attn_weight:
        # q, k, v are each hidden_dim size matrices
        # each matrix has the qs, ks, and vs for each attention head
        qkv_mats = weight.chunk(3 * num_heads, dim=-1)
        sigmas = []
        us = []
        # for attn, compute each of 3 x num_heads singular values independently
        for i, h_mat in enumerate(qkv_mats):
            s, u_ = do_power_iteration(h_mat, u[i])
            sigmas.append(s)
            us.append(u_)
        return torch.stack(sigmas, dim=0), torch.stack(us, dim=0)
    elif is_attn_proj:
        # projection matrix is num_heads x hidden_dim size
        mats = weight.chunk(num_heads, dim=-1)
        sigmas = []
        us = []
        # for attn_proj compute num_heads singular values independently
        for i, h_mat in enumerate(mats):
            s, u_ = do_power_iteration(h_mat, u[i])
            sigmas.append(s)
            us.append(u_)
        return torch.stack(sigmas, dim=0), torch.stack(us, dim=0)
    else:
        return do_power_iteration(weight, u)


def train_model_track_memorization_per_training_set(
    model,
    train_datasets,
    test_dataloaders,
    noise_test_dataloaders,
    noise_data,
    clean_data_corresponding_to_noise,
    num_epochs=num_epochs,
    prompt_len=50,
    k=50,
    PATH="/home/arham23/projects/memorization/model_ckpts/",
    name_of_ckpt="ckpt",
    n_layers=1,
    lam=0.01,
):
    model.train()

    data = torch.cat(
        train_datasets, dim=0
    )  # train_datasets has to be a tuple of datasets
    # create dataloaders (w/ noise and clean data)
    train_dataloader = DataLoader(data, batch_size=batch_size, shuffle=True)

    train_losses = []
    test_losses = []
    noise_test_losses = []
    train_memorized = []
    train_accuracies = []
    test_accuracies = []
    noise_accuracies = []
    percent_memorized = []
    for i in range(len(test_dataloaders)):
        test_losses.append([])  # add empty list to test losses for each test set
        test_accuracies.append([])  # add empty list to test losses for each test set

    for i in range(len(noise_test_dataloaders)):
        noise_test_losses.append([])
        noise_accuracies.append([])

    for i in range(len(train_datasets)):
        train_memorized.append(
            []
        )  # add empty list to train memorized for each subset of trianing

    # init singular vector approximations as random gaussian
    Us = {}
    for name, weight in model.named_parameters():
        if should_compute_sigma(name):
            is_attn_weight = "attn.c_attn" in name
            is_attn_proj = "attn.c_proj" in name
            Us[name] = init_power_vector(
                weight,
                is_attn_weight=is_attn_weight,
                is_attn_proj=is_attn_proj,
                num_heads=NUM_HEADS,
            ).to(device)

    model_checkpoints = []
    checkpoint_epochs = []
    n_power_iterations = 1
    eps = 1e-12
    for epoch in tqdm.tqdm(range(num_epochs), leave=True):
        avg_train_loss = 0
        avg_train_accuracy = 0

        for i, batch in enumerate(
            tqdm.tqdm(train_dataloader, desc="Train Batch", leave=True)
        ):
            model_output = model(batch, labels=batch)
            train_logits = model_output.logits
            train_loss = model_output.loss
            # compute spectral regularizer
            reg_loss = None
            for name, weight in model.named_parameters():
                if should_compute_sigma(name):
                    u = Us[name]
                    is_attn_weight = "attn.c_attn" in name
                    is_attn_proj = "attn.c_proj" in name
                    sigmas, u_ = power_iteration(
                        weight,
                        u,
                        is_attn_weight=is_attn_weight,
                        is_attn_proj=is_attn_proj,
                        num_heads=NUM_HEADS,
                    )
                    Us[name] = u_
                    sum_sigma = torch.sum(sigmas)
                    if reg_loss is None:
                        reg_loss = sum_sigma
                    else:
                        reg_loss += sum_sigma
            # add regularization term to loss
            train_loss += (lam / 2) * reg_loss
            train_loss.backward(retain_graph=True)
            avg_train_loss += train_loss.cpu().item()
            avg_train_accuracy += accuracy(batch, train_logits)
            optimizer.step()
            optimizer.zero_grad()

        train_losses.append((avg_train_loss / len(train_dataloader)))
        train_accuracies.append((avg_train_accuracy.cpu() / len(train_dataloader)))
        # model_alphas.append(get_alpha(model=model))

        with torch.inference_mode():
            # iteration through various train datasets to track memorization
            # for i in range(len(train_datasets)):
            #  dataloader = DataLoader(train_datasets[i], batch_size=batch_size, shuffle=True)
            percent_memorized.append(
                refined_check_percent_memorized(
                    noise_dataset=noise_data,
                    clean_data_set_for_noise=clean_data_corresponding_to_noise,
                    prompt_len=prompt_len,
                    k=k,
                    batch_size=1000,
                    model=model,
                ).cpu()
            )

            # iterate through various test datasets
            for i in range(len(test_dataloaders)):
                avg_test_loss = 0
                avg_test_accuracy = 0
                for batch in test_dataloaders[i]:
                    model_output = model(batch, labels=batch)
                    test_logits = model_output.logits
                    test_loss = model_output.loss
                    avg_test_loss += test_loss.cpu().item()
                    avg_test_accuracy += accuracy(batch, test_logits)
                test_losses[i].append((avg_test_loss / len(test_dataloaders[i])))
                test_accuracies[i].append(
                    (avg_test_accuracy.cpu() / len(test_dataloaders[i]))
                )

            for i in range(len(noise_test_dataloaders)):
                avg_test_loss = 0
                avg_test_accuracy = 0
                for batch in noise_test_dataloaders[i]:
                    model_output = model(batch, labels=batch)
                    test_logits = model_output.logits
                    test_loss = model_output.loss
                    avg_test_loss += test_loss.cpu().item()
                    avg_test_accuracy += accuracy(batch, test_logits)
                noise_test_losses[i].append(
                    (avg_test_loss / len(noise_test_dataloaders[i]))
                )
                noise_accuracies[i].append(
                    (avg_test_accuracy.cpu() / len(noise_test_dataloaders[i]))
                )

        if ((epoch + 1) % checkpoint_every) == 0:
            # Add checkpointing back in
            # checkpoint_epochs.append(epoch)
            # model_checkpoints.append(copy.deepcopy(model.state_dict()))
            MODEL_PATH = (
                PATH
                + f"{name_of_ckpt}_{n_layers}_layer_{epoch+1}_epoch_no_dup_spectralreg_{lam}.pth"
            )
            torch.save(model.state_dict(), MODEL_PATH)
            print(f"Epoch {epoch} Train Loss {train_loss.item()}")
            print(" ")
            print("% mem: ", percent_memorized[-1])
            for test_loss in test_losses:
                print("test loss: ", test_loss[-1])

    return (
        model,
        train_losses,
        test_losses,
        noise_test_losses,
        train_accuracies,
        test_accuracies,
        noise_accuracies,
        percent_memorized,
    )


# Experiments
if __name__ == "__main__":

    # set up arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--n_layers",
        type=int,
        default=1,
        help="The number of layers you want in your toy model.",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.01,
        help="The regularization coefficient for the regularizer term in our loss function.",
    )

    args = parser.parse_args()
    n_layers = args.n_layers
    lam = args.lam

    # Make the data
    # generate indexes for noise vs clean data
    print("generating indices...")
    idxs = list(range(20000 - num_test))
    noise_idxs = sample(idxs, 1000)
    clean_idxs = list(set(idxs) - set(noise_idxs))

    # Mix clean and noise data
    print("mixing distributions...")
    list_of_functions = [seven_function]
    list_of_dataset_sizes = [20000]

    clean_train_dataloader, clean_test_dataloaders = create_data_distributions(
        list_of_functions,
        list_of_dataset_sizes,
        test_set_size=num_test,
        shuffle=True,
        noise=False,
        noise_range=1,
        length=100,
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
        length=100,
    )

    # combine train_dataloaders
    print("combining dataloaders...")
    clean_data = clean_train_dataloader.dataset
    noise_data = noise_train_dataloader.dataset

    # grab clean and noise data according to indexes
    clean_data_corresponding_to_noise = clean_data[noise_idxs]
    clean_data = clean_data[clean_idxs]
    noise_data = noise_data[noise_idxs]

    # Make 4 additional sets of clean data
    # list_of_functions = [two_function, three_function, four_function, five_function]
    # list_of_dataset_sizes = [20000, 20000, 20000, 20000]
    # extra_train_dataloader, extra_test_dataloaders = create_data_distributions(list_of_functions, list_of_dataset_sizes, test_set_size=num_test, shuffle=True, noise=False, noise_range=1, length=100)

    # Need to grab
    train_datasets = (noise_data, clean_data)  # , extra_train_dataloader.dataset)
    # train_datasets += tuple(extra_train_dataloader.dataset)

    # combine test dataloaders
    print("combining test loaders")
    # clean_test_dataloaders += extra_test_dataloaders
    train_datasets = (noise_data, clean_data)  # , extra_train_dataloader.dataset)

    # Count how many noised sequences we have at each prompt length
    print("counting noised sequences...")
    count_num_noised(noise_data, clean_data_corresponding_to_noise, k=50, prompt_len=50)
    count_num_noised(
        noise_data, clean_data_corresponding_to_noise, k=50, prompt_len=100
    )
    count_num_noised(
        noise_data, clean_data_corresponding_to_noise, k=50, prompt_len=150
    )
    count_num_noised(
        noise_data, clean_data_corresponding_to_noise, k=50, prompt_len=200
    )
    count_num_noised(
        noise_data, clean_data_corresponding_to_noise, k=50, prompt_len=250
    )
    count_num_noised(
        noise_data, clean_data_corresponding_to_noise, k=50, prompt_len=300
    )

    # Initializing a model (with random weights) from the configuration
    configuration = GPT2Config(
        vocab_size=14,
        n_layer=n_layers,  # 1,2,4,8,16
        n_head=NUM_HEADS,
        n_embd=128,
        n_positions=max_ctx,
        bos_token_id=10,
        eos_token_id=11,
        use_cache=False,
        hidden_states=False,
        output_attentions=False,
        activation_function="relu",
        attn_pdrop=0,
        resid_pdrop=0,
        embd_pdrop=0,
        initializer_range=0.8 / math.sqrt(128),
    )

    model = GPT2LMHeadModel(configuration)
    model.to(device)

    # Set up optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=wd, betas=betas
    )

    # Train model
    num_epochs = 200
    model.train()
    (
        model,
        train_losses,
        test_losses,
        noise_losses,
        train_accuracies,
        test_accuracies,
        noise_accuracies,
        percent_memorized,
    ) = train_model_track_memorization_per_training_set(
        model,
        train_datasets,
        clean_test_dataloaders,
        noise_test_dataloaders,
        noise_data,
        clean_data_corresponding_to_noise,
        num_epochs=num_epochs,
        name_of_ckpt="5_data_distributions",
        n_layers=n_layers,
        lam=lam,
    )

    # Plot Results
    fig, axs = plt.subplots(1, 3)
    colors = {"train": "blue", "clean": "orange", "noise": "green"}
    print(len(train_losses), len(test_losses), len(percent_memorized), num_epochs)
    plt_line(
        axs[0],
        y_vals=train_losses,
        x_vals=range(num_epochs),
        title="Loss",
        x_label="Epoch",
        y_label="Loss",
        label="train_loss",
    )
    axs[0].plot(
        test_losses[0],
        range(num_epochs),
        label="test_loss_7_clean",
        color=colors["clean"],
    )
    axs[0].plot(
        noise_losses[0],
        range(num_epochs),
        label="test_loss_7_noise",
        color=colors["noise"],
    )

    plt_line(
        axs[1],
        y_vals=train_accuracies,
        x_vals=range(num_epochs),
        title="Accuracies",
        x_label="Epoch",
        y_label="Accuracy",
        label="train_acc",
    )
    axs[1].plot(
        test_accuracies[0],
        range(num_epochs),
        label="test_acc_7_clean",
        color=colors["clean"],
    )
    axs[1].plot(
        noise_accuracies[0],
        range(num_epochs),
        label="test_acc_7_noise",
        color=colors["noise"],
    )

    plt_line(
        axs[2],
        y_vals=percent_memorized,
        x_vals=range(num_epochs),
        title="Memorization",
        x_label="Epoch",
        y_label="% Memorized",
        label="percent_memorized_7_noise",
    )

    for i in range(len(axs)):
        axs[i].grid()
        axs[i].legend()

    plt.savefig(f"n_layers={n_layers}_lam={lam}_spectralregrun.png")
