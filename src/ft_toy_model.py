import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import numpy as np
from utils.dropper import LossDropper
from utils.spectral_reg import *
from src.data.old_data import *
from src.data.IndexedDataset import IndexedDataset
from src.localize.neuron.neuron_utils import refined_check_percent_memorized

import tqdm
import copy
import argparse
import glob
import os
import time


import sys
from random import randrange, choices, sample
from operator import add
import random
import os

import matplotlib.pyplot as plt


def plt_line(
    y_vals, x_val, labels, title="Losses", x_label="losses", y_label="Epoch", path=""
):
    plt.clf()
    for y, label in zip(y_vals, labels):
        plt.plot(x_val, y, label=label)

    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.title(title)
    plt.grid()
    plt.legend()
    plt.savefig(f"{path}{title}.pdf")
    plt.show()
    return 0


"""## config"""

num_test = 1000
betas = (0.9, 0.98)
DATA_SEED = 598

"""## GPT2 small config for model"""

from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel
from models.gpt2_dropout import GPT2LMHeadModel as GPT2LMHeadModelWithDropout
import math


def clm_loss_fn(inputs, logits):
    # Shift so that tokens < n predict n
    shift_labels = inputs[..., 1:].contiguous()
    shift_logits = logits[..., :-1, :].contiguous()
    # Calculate per-token loss
    loss_fct = CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
    # Resize and average loss per sample
    loss_per_sample = loss.view(shift_logits.size(0), shift_logits.size(1)).mean(axis=1)

    return (loss_per_sample).mean(), loss_per_sample


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


def generate(model, input, max_ctx=650, print_output=True):
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


def train_model_track_memorization_per_training_set(
    model,
    train_datasets,
    test_dataloaders,
    noise_data,
    clean_data_corresponding_to_noise,
    dup_idxs,
    num_epochs=200,
    prompt_len=50,
    k=50,
    ckpt_dir="/grand/SuperBERT/mansisak/memorization/model_ckpts/",
    n_layers=1,
    max_ctx=650,
    trigger=100,
    backdoor=0,
    data_name="mult",
    **extra_kwargs,
):
    model.train()

    print(type(train_datasets))
    data = torch.cat(
        train_datasets, dim=0
    )  # train_datasets has to be a tuple of datasets

    """
    if args.ft:
        data = torch.cat(
            (clean_data_corresponding_to_noise,), dim=0
        )  # train_datasets has to be a tuple of datasets
        # data = torch.cat( # Everything except noise
        #    train_datasets[1:], dim=0
        # )  # train_datasets has to be a tuple of datasets
    # create dataloaders (w/ noise and clean data)
    """

    # allows us to index individual examples, useful for example-tied dropout
    # dataloader will automatically give a batched tensor of indices with the correct permutations applied
    indexed_data = IndexedDataset(data)
    data_len = data.shape[0]
    train_dataloader = DataLoader(
        indexed_data, batch_size=args.batch_size, shuffle=True
    )

    train_perplexities = []
    test_perplexities = []
    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    percent_memorized = []
    percent_non_memorized = []
    for i in range(len(test_dataloaders)):
        test_losses.append([])  # add empty list to test losses for each test set
        test_perplexities.append([])  # add empty list to test losses for each test set
        test_accuracies.append([])  # add empty list to test losses for each test set

    for i in range(len(dup_idxs)):
        percent_memorized.append(
            []
        )  # add empty list to perc mem for each duplication set e.g. 10^0, 10^1, ...
        percent_non_memorized.append(
            []
        )  # add empty list to perc mem for each duplication set e.g. 10^0, 10^1, ...

    l1_lam = extra_kwargs.get("l1_lam", 0.0)
    do_dropout = extra_kwargs.get("dropout")

    # Init Loss Truncation if desired
    dropper = None
    if extra_kwargs.get("truncate_loss"):
        dropc = extra_kwargs.get("dropc", 0.4)
        assert dropc >= 0 and dropc <= 1, "dropc parameter must be in the range [0,1]"
        dropper = LossDropper(dropc=dropc, verbose=False)

    # Init for Spectral Regularization if desired
    if do_spectral_reg := extra_kwargs.get("spectral_reg"):
        lam = extra_kwargs.get("lam", 0.01)
        Us = {}
        for name, weight in model.named_parameters():
            if should_compute_sigma(name):
                is_attn_weight = "attn.c_attn" in name
                is_attn_proj = "attn.c_proj" in name
                Us[name] = init_power_vector(
                    weight,
                    is_attn_weight=is_attn_weight,
                    is_attn_proj=is_attn_proj,
                    num_heads=4,
                ).to(device)

    # Automatically find the checkpoint if it exists
    finished_epochs = -1
    if args.ckpt_dir:
        list_of_files = glob.glob(
            f"{args.ckpt_dir}/*.pth"
        )  # * means all if need specific format then *.csv
        if list_of_files:
            latest_file = max(list_of_files, key=os.path.getctime)
            print("latest checkpoint: ", latest_file)
            ckpt = torch.load(latest_file, map_location=torch.device("cpu"))
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            finished_epochs = ckpt["epoch"]
            train_losses = ckpt["train_losses"]
            test_losses = ckpt["test_losses"]
            train_accuracies = ckpt["train_accuracies"]
            test_accuracies = ckpt["test_accuracies"]
            percent_memorized = ckpt["percent_memorized"]
            if "train_perplexities" in ckpt:
                train_perplexities = ckpt["train_perplexities"]
                test_perplexities = ckpt["test_perplexities"]

    for epoch in tqdm.tqdm(range(num_epochs + 1)):
        # make sure
        model.train()

        if epoch <= finished_epochs:
            print("epoch finished: ", epoch)
            continue

        start_time = time.time()
        print("epoch starting: ", epoch)
        avg_train_loss = 0
        avg_train_accuracy = 0
        avg_train_perp = 0

        for batch, example_indices in train_dataloader:
            batch = batch.to(device)
            model_output = None
            if do_dropout:
                model_output = model(batch, labels=batch, input_idx=example_indices)
            else:
                model_output = model(batch, labels=batch)

            train_logits = model_output.logits
            train_loss = model_output.loss

            # apply loss truncation
            if dropper is not None:
                # print("Train_loss mean: ", train_loss)
                computed_mean_loss, train_loss = clm_loss_fn(batch, train_logits)
                # print("Computed Train_loss mean: ", train_loss)
                # train_loss.view(-1, batch_size)
                # train_loss = train_loss.mean(dim=0)  # aggregate by sequence
                mask = dropper(
                    train_loss
                )  # The dropper returns a mask of 0s where data should be dropped.
                train_loss *= mask  # Mask out the high losses
                train_loss = train_loss.mean()  # Aggregate

            # apply spectral reg
            if do_spectral_reg:
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
                            num_heads=4,
                        )
                        Us[name] = u_
                        sum_sigma = torch.sum(sigmas)
                        if reg_loss is None:
                            reg_loss = sum_sigma
                        else:
                            reg_loss += sum_sigma
                # add regularization term to loss
                train_loss += (lam / 2) * reg_loss

            # apply L1 Regularization
            if l1_lam != 0.0:
                all_params = torch.cat([x.view(-1) for x in model.parameters()])
                l1_norm = l1_lam * torch.norm(all_params, 1)
                train_loss += l1_lam * l1_norm

            train_loss.backward()
            avg_train_loss += train_loss.cpu().item()
            avg_train_perp += torch.exp(train_loss).cpu().item()
            avg_train_accuracy += accuracy(batch, train_logits)
            optimizer.step()
            optimizer.zero_grad()
        end_time = time.time()
        print("Time for epoch: ", end_time - start_time)

        train_losses.append((avg_train_loss / len(train_dataloader)))
        train_accuracies.append((avg_train_accuracy.cpu() / len(train_dataloader)))
        train_perplexities.append((avg_train_perp / len(train_dataloader)))
        # model_alphas.append(get_alpha(model=model))

        if ((epoch) % args.checkpoint_every) == 0:
            # make sure
            model.eval()
            print("saving ckpt")

            with torch.inference_mode():
                # iteration through various train datasets to track memorization
                # for i in range(len(train_datasets)):
                #  dataloader = DataLoader(train_datasets[i], batch_size=batch_size, shuffle=True)
                for i in range(len(dup_idxs)):
                    idxs = dup_idxs[i]
                    n_data = noise_data[idxs]
                    c_data = clean_data_corresponding_to_noise[idxs]
                    if backdoor:
                        n_data = test_dataloaders[-2].dataset  # backdoor trig data
                        c_data = test_dataloaders[
                            -1
                        ].dataset  # backdoor trig data w/o trig behavior
                    percent_mem, percent_non_mem, mem_seq, clean_mem_seq = (
                        refined_check_percent_memorized(
                            noise_dataset=n_data,
                            clean_data_set_for_noise=c_data,
                            prompt_len=prompt_len,
                            k=k,
                            batch_size=32,
                            model=model,
                            max_ctx=max_ctx,
                            pad_token_id=pad_token_id,
                            backdoor=backdoor,
                            trigger=trigger,
                            data_name=data_name,
                        )
                    )
                    percent_memorized[i].append(percent_mem)
                    percent_non_memorized[i].append(percent_non_mem)

                # iterate through various test datasets
                for i in range(len(test_dataloaders)):
                    avg_test_loss = 0
                    avg_test_perp = 0
                    avg_test_accuracy = 0
                    for batch in test_dataloaders[i]:
                        model_output = model(batch, labels=batch)
                        test_logits = model_output.logits
                        test_loss = model_output.loss
                        avg_test_loss += test_loss.cpu().item()
                        avg_test_perp += torch.exp(test_loss).cpu().item()
                        avg_test_accuracy += accuracy(batch, test_logits)
                    test_losses[i].append((avg_test_loss / len(test_dataloaders[i])))
                    test_accuracies[i].append(
                        (avg_test_accuracy.cpu() / len(test_dataloaders[i]))
                    )
                    test_perplexities[i].append(
                        (avg_test_perp / len(test_dataloaders[i]))
                    )

            if not os.path.exists(ckpt_dir):
                os.makedirs(ckpt_dir)
            MODEL_PATH = f"{ckpt_dir}/{n_layers}_layer_{epoch}_epoch.pth"
            if args.ft:
                if not os.path.exists(f"{ckpt_dir}/ft/"):
                    os.makedirs(f"{ckpt_dir}/ft/")
                MODEL_PATH = f"{ckpt_dir}/ft/{n_layers}_layer_{epoch}_epoch.pth"

            print("Model path: ", MODEL_PATH)
            # Add checkpointing back in
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "train_accuracies": train_accuracies,
                    "test_accuracies": test_accuracies,
                    "train_losses": train_losses,
                    "test_losses": test_losses,
                    "train_perplexities": train_perplexities,
                    "test_perplexities": test_perplexities,
                    "percent_memorized": percent_memorized,
                    "percent_non_mem": percent_non_memorized,
                },
                MODEL_PATH,
            )

            if args.plot == 1:
                print("making plots")
                plt_line(
                    [
                        train_losses,
                        test_losses[0],
                        test_losses[1],
                        test_losses[2],
                        test_losses[3],
                        test_losses[4],
                    ],
                    x_val=np.arange(0, len(train_losses), 1),
                    labels=[
                        "train_loss",
                        "test_loss_7",
                        "test_loss_2",
                        "test_loss_3",
                        "test_loss_4",
                        "test_loss_5",
                    ],
                    title=f"Losses {n_layers}",
                    x_label="Epoch",
                    y_label="Loss",
                    path=ckpt_dir + "/",
                )
                plt_line(
                    [
                        train_accuracies,
                        test_accuracies[0],
                        test_accuracies[1],
                        test_accuracies[2],
                        test_accuracies[3],
                        test_accuracies[4],
                        test_accuracies[1],
                    ],
                    x_val=np.arange(0, len(train_losses), 1),
                    labels=[
                        "train_acc",
                        "test_acc_7",
                        "test_acc_2",
                        "test_acc_3",
                        "test_acc_4",
                        "test_acc_5",
                    ],
                    title=f"Accuracies {n_layers}",
                    x_label="Epoch",
                    y_label="Accuracy",
                    path=ckpt_dir + "/",
                )
                plt_line(
                    [percent_memorized],
                    x_val=np.arange(0, len(train_losses), 1),
                    labels=["percent_memorized_7_noise"],
                    title=f"Memorization {n_layers}",
                    x_label="Epoch",
                    y_label="% Memorized",
                    path=ckpt_dir + "/",
                )
            # MODEL_PATH = PATH + f"{n_layers}_layer_{epoch+1}_epoch_no_dup.pth"
            # torch.save(model.state_dict(), "just_model.pt")
            print(f"Epoch {epoch}")
            print(f"Train Loss {train_loss.item()}")
            print(" ")
            for perc_mem in percent_memorized:
                print("% mem: ", perc_mem[-1])
            for test_loss in test_losses:
                print("test loss: ", test_loss[-1])
            for test_acc in test_accuracies:
                print("test acc: ", test_acc[-1])
            for test_perp in test_perplexities:
                print("test perp: ", test_perp[-1])

    return (
        model,
        train_losses,
        test_losses,
        train_accuracies,
        test_accuracies,
        percent_memorized,
    )


# Experiments
if __name__ == "__main__":
    # set up arg parser
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--plot",
        type=int,
        default=0,
        help="Save plots (true or false)",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning Rate for training.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=128,
        help="Batch Size for training.",
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=14,
        help="Number of tokens in model vocab.",
    )
    parser.add_argument(
        "--n_layers",
        type=int,
        default=1,
        help="The number of layers you want in your toy model.",
    )
    parser.add_argument(
        "--truncate_loss",
        action="store_true",
        help="Whether to apply loss truncation during training.",
    )
    parser.add_argument(
        "--dropc",
        type=float,
        default=0.4,
        help="If loss truncation is enabled, what fraction of the data to drop. Should be in [0,1].",
    )
    parser.add_argument(
        "--spectral_reg",
        action="store_true",
        help="Whether to apply spectral regularization during training.",
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.01,
        help="The regularization coefficient for the spectral regularization term in our loss function.",
    )
    parser.add_argument(
        "--example_tied_dropout",
        action="store_true",
        help="Whether to apply example-tied dropout during training.",
    )
    parser.add_argument(
        "--p_mem",
        type=float,
        default=0.1,
        help="The fraction of dropped neurons for the example-tied-droupout regularization strategy.",
    )
    parser.add_argument(
        "--l1-reg",
        type=float,
        default=0.0,
        help="Regularization coefficient for L1 Regularization (Lasso Reg.)",
    )
    parser.add_argument(
        "--l2-reg",
        type=float,
        default=0.1,
        help="Regularization coefficient for weight decay (L2 Reg./Ridge Reg.)",
    )
    parser.add_argument(
        "--checkpoint_every",
        type=int,
        default=5,
        help="The number of epochs between each checkpoint.",
    )
    parser.add_argument(
        "--max_ctx",
        type=int,
        default=650,
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
        default=20000,
        help="Number of points from the 2 distribution.",
    )
    parser.add_argument(
        "--num_3",
        type=int,
        default=20000,
        help="Number of points from the 3 distribution.",
    )
    parser.add_argument(
        "--num_4",
        type=int,
        default=20000,
        help="Number of points from the 4 distribution.",
    )
    parser.add_argument(
        "--num_5",
        type=int,
        default=20000,
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
        default=100,
        help="Amount of numbers in each math sequence",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for dataset generation.",
    )
    parser.add_argument(
        "--ft",
        type=int,
        default=0,
        help="Fine tune model w/ clean data corresponding to noise.",
    )
    parser.add_argument(
        "--epochs", type=int, default=200, help="The number of epochs for training."
    )
    parser.add_argument(
        "--ckpt_dir", type=str, default="ckpts", help="Name of the ckpts parent folder."
    )
    parser.add_argument(
        "--duplicate",
        type=int,
        default=0,
        help="Whether or not to do duplication on dataset.",
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
            "wiki",
            "wiki_fast",
            "shakespeare",
            "increment",
            "mult",
            "exp",
            "exponential",
            "increment_3",
            "mult_3",
            "exp_3",
            "exponential_3",
            "increment_5",
            "mult_5",
            "exp_5",
            "exponential_5",
        ],
        type=str,
        default="increment",
        help="Name of function type you want to train with.",
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    extra_kwargs = {
        "truncate_loss": args.truncate_loss,
        "dropc": args.dropc,
        "spectral_reg": args.spectral_reg,
        "lam": args.lam,
        "dropout": args.example_tied_dropout,
        "l1_lam": args.l1_reg,
    }

    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda":
        print("DEVICE: ", device, "name: ", torch.cuda.get_device_name(device=device))

    # Make the data
    print("Generating data...")
    data_path = f"data/{args.data_name}_{args.num_7}_{args.num_2}_{args.num_3}_{args.num_4}_{args.num_5}_data_{args.length}_{args.num_test}_{args.num_noise}_{args.max_ctx}_{args.seed}.pt"
    pad_token_id = 13
    bos_token_id = 10
    eos_token_id = 11
    if args.data_name in ("shakespeare", "wiki", "wiki_fast"):
        data_path = f"data/{args.data_name}_{args.max_ctx}_{args.seed}.pt"
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
        data_path = data_path[:-3]
        data_path = f"{data_path}_dup.pt"
    print(data_path)

    data_test = get_data(
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
        batch_size=args.batch_size,
    )
    print("data len: ", len(data_test))

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
        batch_size=args.batch_size,
    )
    if args.backdoor:
        dup_idxs = [list(range(len(clean_test_dataloaders[-2].dataset)))]
    print("COUNTING FROM GENERTED DATA")
    print("Noise data shape: ", noise_data.shape)
    print(
        "clean_data_correspoinding_to_noise data shape: ",
        clean_data_corresponding_to_noise.shape,
    )

    # Count how many noised sequences we have at each prompt length
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

    # Need to have significantly fewer noised samples in the dataset and track accuracy and memorization on them separatly
    # Now we are going to be more strict with how we measure memorization

    # Initializing a model (with random weights) from the configuration
    # TODO: fix bos and eos token ID for training
    configuration = GPT2Config(
        vocab_size=args.vocab_size,
        n_layer=args.n_layers,  # 1,2,4,8,16
        n_head=4,
        n_embd=args.n_embed,
        n_positions=args.max_ctx,
        bos_token_id=bos_token_id,
        eos_token_id=eos_token_id,
        pad_token_id=pad_token_id,
        use_cache=False,
        hidden_states=False,
        output_attentions=False,
        activation_function="relu",
        attn_pdrop=0,
        resid_pdrop=0,
        embd_pdrop=0,
        initializer_range=0.8 / math.sqrt(args.n_embed),  # 0.8 / sqrt(d_model)
    )

    # change data_len based on FT or not
    data = torch.cat(
        train_datasets, dim=0
    )  # train_datasets has to be a tuple of datasets
    data_len = data.shape[0]
    if args.ft:
        # TODO (MS): fix!!
        data_len = clean_data_corresponding_to_noise.shape[0]

    print(configuration)
    print("data len: ", data_len)
    model = None
    if args.example_tied_dropout:
        model = GPT2LMHeadModelWithDropout(configuration, data_len, args.p_mem)
    else:
        model = GPT2LMHeadModel(configuration)

    model.to(device)

    # Set up optimizer
    weight_decay = args.l2_reg
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=weight_decay, betas=betas
    )

    # Train model
    # TODO (MS): implement distributed training
    model.train()
    (
        model,
        train_losses,
        test_losses,
        train_accuracies,
        test_accuracies,
        percent_memorized,
    ) = train_model_track_memorization_per_training_set(
        model,
        train_datasets,
        clean_test_dataloaders,
        noise_data,
        clean_data_corresponding_to_noise,
        dup_idxs,
        num_epochs=args.epochs,
        ckpt_dir=args.ckpt_dir,
        n_layers=args.n_layers,
        max_ctx=args.max_ctx,
        trigger=trigger,
        backdoor=args.backdoor,
        data_name=args.data_name,
        **extra_kwargs,
    )