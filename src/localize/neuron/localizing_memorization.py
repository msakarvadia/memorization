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


"""# Activations"""


def register_hook(model, layer_idx, ori_activations, attr_str):
    ff_layer = get_attributes(model, attr_str)

    # print(attr_str)
    # print("input shape in hook: ", ori_activations.shape)

    def hook_fn(m, i, o):
        # print(ori_activations.shape)
        # print("o dim: ", o.shape)
        ori_activations[layer_idx] = o.squeeze().cpu()
        # ori_activations[layer_idx] = o.cpu()

    return ff_layer.register_forward_hook(hook_fn)


@torch.no_grad()
def get_ori_activations_ACT(inner_dim, model, inputs):
    # seq_len = inputs['input_ids'].shape[1]
    seq_len = inputs.shape[1]
    batch_size = inputs.shape[0]
    ori_activations = torch.zeros(
        (model.config.n_layer, batch_size, seq_len, inner_dim)
    )

    handles = []
    for ly in range(model.config.n_layer):
        handle = register_hook(
            model,
            ly,
            ori_activations,
            f"{model.attr_dict['transformer_layer']}.{ly}.{model.attr_dict['ffn_act']}",
        )
        handles.append(handle)

    # print("input shape: ", inputs.shape)
    out = model(inputs)
    # print("here")
    for handle in handles:  # detach the hooks
        handle.remove()

    return ori_activations


def largest_act(inner_dim, model, inputs, gold_set, model_name="gpt2", prompt_len=50):

    @torch.no_grad()
    def get_ffn_norms():
        all_norms = torch.zeros((model.config.n_layer, inner_dim))
        for ly in range(model.config.n_layer):
            attr_str = f"{model.attr_dict['transformer_layer']}.{ly}.{model.attr_dict['ffn_out']}.weight"
            weights = get_attributes(model, attr_str)
            if "gpt2" in model_name:
                norms = torch.norm(weights, dim=1)
            else:
                norms = torch.norm(weights, dim=0)
            all_norms[ly] = norms.cpu()

        return all_norms

    all_norms = get_ffn_norms()
    # print("norms shape: ", all_norms.shape)

    # prompt_start_i = args.prompt_len -1 if hasattr(args, 'prompt_len') else 0  # -1 for 0-indexed
    prompt_start_i = prompt_len - 1

    activations = get_ori_activations_ACT(inner_dim, model, inputs)
    # print(activations.shape)
    activations = activations[
        :, :, prompt_start_i:-1
    ]  # [n_layer, suffix_len, inner_dim]
    all_norms = get_ffn_norms()

    # print(activations.shape)
    # print(activations.mean(1).shape)

    act_mean = (
        activations.mean(1).mean(1).cpu().abs() * all_norms
    )  # Average once over batch size, then again over suffix length
    # torch.save(act_mean, os.path.join(args.out_dir, 'act-mean.pt'))
    # if gold_set is not None:
    #    score = get_layerwise_scores(act_mean, gold_set, args.ratio)
    return act_mean


def remove_all_forward_hooks(model: torch.nn.Module) -> None:
    for name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks: Dict[int, Callable] = OrderedDict()
            remove_all_forward_hooks(child)


"""# Integrated Gradients"""


@torch.no_grad()
def get_ori_activations_IG(inner_dim, model, inputs):
    # seq_len = inputs['input_ids'].shape[1]
    seq_len = inputs.shape[1]
    ori_activations = torch.zeros((model.config.n_layer, seq_len, inner_dim))

    handles = []
    for ly in range(model.config.n_layer):
        handle = register_hook(
            model,
            ly,
            ori_activations,
            f"{model.attr_dict['transformer_layer']}.{ly}.{model.attr_dict['ffn_act']}",
        )
        handles.append(handle)

    out = model(inputs)

    for handle in handles:  # detach the hooks
        handle.remove()

    return ori_activations


def scaled_input(activations, steps, device):
    """
    Tiles activations along the batch dimension - gradually scaling them over
    `steps` steps from 0 to their original value over the batch dimensions.
    """
    tiled_activations = activations.expand((steps, len(activations)))
    scales = torch.linspace(start=0, end=1, steps=steps)[:, None]  # (steps, 1)
    out = (tiled_activations * scales).to(device)
    return out  # [steps, inner_dim]


def integrated_gradients(
    inner_dim, model, inputs, gold_set, ig_steps, device, n_batches=16, prompt_len=50
):
    activations = get_ori_activations_IG(inner_dim, model, inputs)

    target_ids = inputs.squeeze()[1:].tolist()
    seq_len = inputs.shape[1]

    n_layer = model.config.n_layer
    # prompt_start_i = args.prompt_len -1 if hasattr(args, 'prompt_len') else 0  # -1 for 0-indexed
    prompt_start_i = prompt_len - 1
    integrated_grads_ = torch.zeros((n_layer, seq_len - 1 - prompt_start_i, inner_dim))

    for ly in tqdm(range(n_layer)):
        integrated_grads = []
        for i in range(prompt_start_i, seq_len - 1):
            ori_activations = activations[ly, i]

            scaled_weights = scaled_input(
                ori_activations, steps=ig_steps, device=device
            )
            scaled_weights.requires_grad_(True)

            ff_attrs = f"{model.attr_dict['transformer_layer']}.{ly}.{model.attr_dict['ffn_act']}"
            integrated_grads_t = torch.zeros(inner_dim)
            for batch_weights in scaled_weights.chunk(n_batches):  # batch ig_steps
                bs = len(batch_weights)
                cur_input_ids = inputs[:, : i + 1].expand(
                    bs, i + 1
                )  # [ig_steps, cur_seq_len]

                # patch the model with the scaled activations
                patch_ff_layer(
                    model,
                    ff_attrs,
                    replacement_activations=batch_weights,
                )

                outputs = model(cur_input_ids)
                probs = torch.nn.functional.softmax(
                    outputs.logits[:, -1, :], dim=-1
                )  # [ig_steps, vocab]
                grad = torch.autograd.grad(
                    torch.unbind(probs[:, target_ids[i]]), batch_weights
                )[
                    0
                ]  # [ig_steps, inner_dim]
                integrated_grads_t += grad.sum(dim=0).cpu()  # sum over ig_steps

                unpatch_ff_layer(
                    model,
                    ff_attrs,
                )
            # Eq 5, 1/m * (ori - baseline) * (\Sum grads), where we use baseline = 0
            integrated_grads_t = ori_activations * integrated_grads_t / ig_steps
            integrated_grads.append(integrated_grads_t)

        integrated_grads_[ly] = torch.stack(integrated_grads, dim=0)

    # print(len(integrated_grads))
    ig_mean = integrated_grads_.mean(1).cpu()
    # torch.save(ig_mean, os.path.join(args.out_dir, 'ig-mean.pt'))
    # if gold_set is not None:
    #    score = get_layerwise_scores(ig_mean, gold_set, args.ratio)
    return ig_mean


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
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
    patched = False

    if not patched:
        patch_hardconcrete(model, model_name, mask_p=0.5, beta=2 / 3)
        patched = True
        model.to(device)
    else:
        if "gpt2" in model_name:  # the newly loaded weights need to be transposed
            transpose_conv1d(model)
        reinit_hardconcrete(model)

    concrete_params = hard_concrete(
        lr=1e-2,
        epoch=100,
        lambda_l1=1000,
        stop_loss=1e-1,
        threshold=1e-1,
        model=model,
        inputs=noise_data,
        gold_set=None,
    )

    batch_size = 1000
    seq_len = noise_data.shape[1]

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

    apply_ablation_mask_to_neurons(concrete_params, model=model, ratio=0.05)

    print("\n AFTER MASKING---------")

    perc_mem, acc, perplex_clean, perplex_noise = track_all_metrics(
        noise_data=noise_data,
        clean_data_corresponding_to_noise=clean_data_corresponding_to_noise,
        clean_test_dataloaders=clean_test_dataloaders,
        model=model,
        prompt_len=50,
        batch_size=1000,
    )

    remove_ablation_mask_from_neurons(model)
    model = get_model(args.model_path, args.n_layers)

    ## Zero-out
    delta_losses = fast_zero_out_vector(
        inner_dim=model.inner_dim,
        ratio=0.001,
        n_batches=16,
        model=model,
        inputs=noise_data,
        labels=clean_data_corresponding_to_noise,
        prompt_len=50,
    )

    # Baseline perplexity for randome sequences
    num_classes = 50000
    bs = 1

    # your model outputs / logits
    output = torch.rand(bs, num_classes)
    print(output.shape)

    # your targets
    target = torch.randint(num_classes, (bs,))
    print(target.shape)

    # getting loss using cross entropy
    loss = torch.nn.functional.cross_entropy(output, target)

    # calculating perplexity
    perp = torch.exp(loss)
    print("Loss:", loss, "PP:", perp)

    batch_size = 1000
    seq_len = noise_data.shape[1]

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
        delta_losses, model=model, inputs=noise_data, ratio=0.01
    )

    print("\n AFTER MASKING---------")

    perc_mem, acc, perplex_clean, perplex_noise = track_all_metrics(
        noise_data=noise_data,
        clean_data_corresponding_to_noise=clean_data_corresponding_to_noise,
        clean_test_dataloaders=clean_test_dataloaders,
        model=model,
        prompt_len=50,
        batch_size=1000,
    )

    remove_ablation_mask_from_neurons(model)

    batch_size = 1000
    seq_len = noise_data.shape[1]

    print("BEFORE MASKING---------")

    perc_mem, acc, perplex_clean, perplex_noise = track_all_metrics(
        noise_data=noise_data,
        clean_data_corresponding_to_noise=clean_data_corresponding_to_noise,
        clean_test_dataloaders=clean_test_dataloaders,
        model=model,
        prompt_len=50,
        batch_size=1000,
    )

    apply_ablation_mask_to_neurons(delta_losses, model=model, ratio=0.5)

    print("\n AFTER MASKING---------")

    perc_mem, acc, perplex_clean, perplex_noise = track_all_metrics(
        noise_data=noise_data,
        clean_data_corresponding_to_noise=clean_data_corresponding_to_noise,
        clean_test_dataloaders=clean_test_dataloaders,
        model=model,
        prompt_len=50,
        batch_size=1000,
    )

    remove_ablation_mask_from_neurons(model)

    batch_size = 1000
    seq_len = noise_data.shape[1]

    print("BEFORE MASKING---------")

    perc_mem, acc, perplex_clean, perplex_noise = track_all_metrics(
        noise_data=noise_data,
        clean_data_corresponding_to_noise=clean_data_corresponding_to_noise,
        clean_test_dataloaders=clean_test_dataloaders,
        model=model,
        prompt_len=50,
        batch_size=1000,
    )

    apply_mean_ablation_mask_to_neurons(
        delta_losses, model=model, inputs=noise_data, ratio=0.25
    )

    print("\n AFTER MASKING---------")

    perc_mem, acc, perplex_clean, perplex_noise = track_all_metrics(
        noise_data=noise_data,
        clean_data_corresponding_to_noise=clean_data_corresponding_to_noise,
        clean_test_dataloaders=clean_test_dataloaders,
        model=model,
        prompt_len=50,
        batch_size=1000,
    )

    remove_ablation_mask_from_neurons(model)

    ## Slimming
    patched = False

    if not patched:
        patch_slim(model)
        patched = True
        model.to(device)  # send the coef_parameters in patch to gpu
    else:
        reinit_slim(model)
    slim_params = slim(
        lr=1e-2,
        epoch=100,
        lambda_l1=1000,
        stop_loss=1e-1,
        threshold=1e-1,
        model=model,
        inputs=noise_data,
        gold_set=None,
    )

    batch_size = 1000
    seq_len = noise_data.shape[1]

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

    apply_ablation_mask_to_neurons(slim_params, model=model, ratio=0.01)

    print("\n AFTER MASKING---------")

    perc_mem, acc, perplex_clean, perplex_noise = track_all_metrics(
        noise_data=noise_data,
        clean_data_corresponding_to_noise=clean_data_corresponding_to_noise,
        clean_test_dataloaders=clean_test_dataloaders,
        model=model,
        prompt_len=50,
        batch_size=1000,
    )

    remove_ablation_mask_from_neurons(model)

    ## Activations
    model = get_model(args.model_path, args.n_layers)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    act_mean = largest_act(
        inner_dim=model.inner_dim,
        model=model,
        inputs=noise_data,
        gold_set=None,
        model_name="gpt2",
        prompt_len=50,
    )
    remove_all_forward_hooks(model)

    batch_size = 1000
    seq_len = noise_data.shape[1]

    print("BEFORE MASKING---------")

    perc_mem, acc, perplex_clean, perplex_noise = track_all_metrics(
        noise_data=noise_data,
        clean_data_corresponding_to_noise=clean_data_corresponding_to_noise,
        clean_test_dataloaders=clean_test_dataloaders,
        model=model,
        prompt_len=50,
        batch_size=1000,
    )

    apply_ablation_mask_to_neurons(act_mean, model=model, ratio=0.1)

    print("\n AFTER MASKING---------")

    perc_mem, acc, perplex_clean, perplex_noise = track_all_metrics(
        noise_data=noise_data,
        clean_data_corresponding_to_noise=clean_data_corresponding_to_noise,
        clean_test_dataloaders=clean_test_dataloaders,
        model=model,
        prompt_len=50,
        batch_size=1000,
    )

    remove_ablation_mask_from_neurons(model)

    remove_all_forward_hooks(model)

    ## Integrated Gradients

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ig_mean = integrated_gradients(
        inner_dim=model.inner_dim,
        model=model,
        inputs=noise_data[0].unsqueeze(0),
        gold_set=None,
        ig_steps=20,
        device=device,
        n_batches=16,
        prompt_len=50,
    )

    # num_layer = 1
    ig_mean = torch.zeros(model.config.n_layer, model.inner_dim)
    num_iters = 10
    for i in range(num_iters):  # Num steps
        # print(i.shape)
        ig_mean += integrated_gradients(
            inner_dim=model.inner_dim,
            model=model,
            inputs=noise_data[i].unsqueeze(0),
            gold_set=None,
            ig_steps=20,
            device=device,
            n_batches=16,
        )

    ig_mean /= num_iters

    batch_size = 1000
    seq_len = noise_data.shape[1]
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

    apply_ablation_mask_to_neurons(ig_mean, model=model, ratio=0.01)

    print("\n AFTER MASKING---------")

    perc_mem, acc, perplex_clean, perplex_noise = track_all_metrics(
        noise_data=noise_data,
        clean_data_corresponding_to_noise=clean_data_corresponding_to_noise,
        clean_test_dataloaders=clean_test_dataloaders,
        model=model,
        prompt_len=50,
        batch_size=1000,
    )

    remove_ablation_mask_from_neurons(model)
