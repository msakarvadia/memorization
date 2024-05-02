import sys
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
import torch
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

model = get_model(False)

"""# Ablation Utility Functions"""


def apply_ablaton_mask_to_neurons(neuron_weightings, model, ratio=0.01):
    print("Num of dropped neurons per layer: ", int(model.inner_dim * ratio // 1))
    for ly in tqdm(range(model.config.n_layer)):
        attr_str = (
            f"{model.attr_dict['transformer_layer']}.{ly}.{model.attr_dict['ffn_act']}"
        )

        coeffs = neuron_weightings[ly]

        val, idx = torch.topk(
            coeffs, k=int(model.inner_dim * ratio // 1)
        )  # grab neuron idxs that have highest diff losses
        # make one hot mask for that
        mask = torch.ones(model.inner_dim)
        mask[idx] = 0

        patch_ff_layer(
            model,
            attr_str,
            # onehot_coef = batch_coef,
            onehot_coef=mask.to(device),
        )

    return model


# apply_ablaton_mask_to_neurons(delta_losses, model=model, ratio=0.01)


def remove_ablation_mask_from_neurons(model):
    for ly in tqdm(range(model.config.n_layer)):
        attr_str = (
            f"{model.attr_dict['transformer_layer']}.{ly}.{model.attr_dict['ffn_act']}"
        )
        unpatch_ff_layer(
            model,
            attr_str,
        )

    return 0


def apply_mean_ablaton_mask_to_neurons(neuron_weightings, model, inputs, ratio=0.01):
    print("Num of dropped neurons per layer: ", int(model.inner_dim * ratio // 1))
    for ly in tqdm(range(model.config.n_layer)):
        attr_str = (
            f"{model.attr_dict['transformer_layer']}.{ly}.{model.attr_dict['ffn_act']}"
        )

        coeffs = neuron_weightings[ly]

        val, idx = torch.topk(
            coeffs, k=int(model.inner_dim * ratio // 1)
        )  # grab neuron idxs that have highest diff losses

        patch_ff_layer(
            model,
            attr_str,
            # onehot_coef = batch_coef,
            mean_ablation_idxs=idx,
            # replacement_activations = activations.to(device),
        )

    return model


# apply_mean_ablaton_mask_to_neurons(delta_losses, model=model, inputs=noise_data, ratio=0.1)
# remove_ablation_mask_from_neurons(model)


def apply_noise_ablaton_mask_to_neurons(
    neuron_weightings, model, inputs, ratio=0.01
):  # TODO check this correctness
    print("Num of dropped neurons per layer: ", int(model.inner_dim * ratio // 1))
    for ly in tqdm(range(model.config.n_layer)):
        attr_str = (
            f"{model.attr_dict['transformer_layer']}.{ly}.{model.attr_dict['ffn_act']}"
        )

        coeffs = neuron_weightings[ly]

        val, idx = torch.topk(
            coeffs, k=int(model.inner_dim * ratio // 1)
        )  # grab neuron idxs that have highest diff losses

        patch_ff_layer(
            model,
            attr_str,
            # onehot_coef = batch_coef,
            noise_ablation_idxs=idx,
            # replacement_activations = activations.to(device),
        )

    return model


# apply_ablaton_mask_to_neurons(delta_losses, model=model, ratio=0.01)
# apply_noise_ablaton_mask_to_neurons(delta_losses, model=model, inputs=noise_data, ratio=0.1)
# remove_ablation_mask_from_neurons(model)

"""# Slimming"""

model = get_model(g_drive=False)

torch.autograd.set_detect_anomaly(False)


def patch_slim(model):
    for ly in range(model.config.n_layer):
        ff_attrs = (
            f"{model.attr_dict['transformer_layer']}.{ly}.{model.attr_dict['ffn_act']}"
        )
        patch_ff_layer(
            model,
            ff_attrs,
            intermediate_size=model.inner_dim,
        )


def reinit_slim(model):
    for ly in range(model.config.n_layer):
        attrs_str = f"{model.attr_dict['transformer_layer']}.{ly}.{model.attr_dict['ffn_act']}.slim_coef"
        coef = get_attributes(model, attrs_str)
        init.ones_(coef)


def compute_l1_loss(model, start_layer_idx):
    l1_loss = 0.0
    N = 0
    cnt = 0
    for m in model.modules():
        if isinstance(m, Patch):
            cnt += 1
            if cnt > start_layer_idx:
                l1_loss = l1_loss + torch.norm(m.slim_coef, 1)
                N += len(m.slim_coef)
    l1_loss /= N
    return l1_loss


def slim(lr, epoch, lambda_l1, stop_loss, threshold, model, inputs, gold_set):
    model.eval()

    # start_layer_idx = args.start_mask_layer if hasattr(args, 'start_mask_layer') else 0
    start_layer_idx = 0
    # set tunable parameters
    cnt = 0
    params = []
    for n, p in model.named_parameters():
        if "slim" in n:
            cnt += 1
            if cnt > start_layer_idx:
                p.requires_grad = True
                print(n)
            else:
                p.requires_grad = False
            params.append(p)
        else:
            p.requires_grad = False
    print("-" * 100)

    # print(params)
    optimizer = torch.optim.Adam(params, lr=lr)

    # training
    scores, reg_losses, lm_losses = [], [], []
    for i in tqdm(range(epoch)):
        optimizer.zero_grad()

        outputs = model(inputs, labels=inputs)
        l1_loss = compute_l1_loss(model, start_layer_idx)
        # print("loss: ", l1_loss)

        lm_loss = outputs.loss
        # print("lm loss: ", lm_loss)
        loss = lm_loss + lambda_l1 * l1_loss

        if (i + 1) % 10 == 0:
            ckpt_params = torch.stack(params).clamp(min=0.0, max=1.0)
            sparsity = (ckpt_params[start_layer_idx:] < threshold).float().mean().item()
            print(
                i + 1, f"lm loss: {lm_loss.item():.3f}, l1 loss: {l1_loss.item():.2f}"
            )
            print("  Sparsity:", sparsity)
            # if gold_set:
            #    score = get_layerwise_scores(ckpt_params, gold_set, args.ratio)
            # else:
            #    score = 0 # dummy
            #    if args.save_ckpt: save_params(args, ckpt_params, f'{i+1}.pt')
            # scores.append(score)
            lm_losses.append(lm_loss.item())
            reg_losses.append(l1_loss.item())
            if l1_loss < stop_loss:
                break

        # print("here")
        # print("------------------")
        loss.backward()
        optimizer.step()

    params = torch.stack(params).clamp(min=0.0, max=1.0).detach().cpu()
    # torch.save(params, os.path.join(args.out_dir, 'slim.pt'))
    # save_records(args, scores, np.array(reg_losses), np.array(lm_losses), sparsity)

    return params


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

model = get_model(False)

print("BEFORE MASKING---------")

perc_mem, acc, perplex_clean, perplex_noise = track_all_metrics(
    noise_data=noise_data,
    clean_data_corresponding_to_noise=clean_data_corresponding_to_noise,
    clean_test_dataloaders=clean_test_dataloaders,
    model=model,
    prompt_len=50,
    batch_size=1000,
)

apply_ablaton_mask_to_neurons(slim_params, model=model, ratio=0.01)

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

slim_params.shape

"""# Implement Localization Strategy: Zero-out

We will implement a few of the strategies from this paper:
 https://arxiv.org/pdf/2311.09060.pdf

 Will need to track memorization before and after localization, and accuracy and perplexity before and after removing memorization on target task.
"""

model_name = "mem_gpt2"
set_model_attributes(model, model_name)
print(model.inner_dim)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@torch.no_grad()
def fast_zero_out_vector(
    inner_dim, ratio, n_batches, model, inputs, labels, prompt_len, gold_set=None
):
    model.eval()
    loss_ori = model(inputs, labels=labels).loss.item()

    losses = torch.zeros((model.config.n_layer, inner_dim))
    seq_len = inputs.shape[1]
    print("seq len: ", seq_len)
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none", ignore_index=-100)

    for ly in tqdm(range(model.config.n_layer)):
        inner_losses = []

        inner_dim = model.inner_dim
        attr_str = (
            f"{model.attr_dict['transformer_layer']}.{ly}.{model.attr_dict['ffn_act']}"
        )

        for zero_idx in range(inner_dim):
            mask = torch.zeros(inner_dim)
            mask[zero_idx] = 1

            patch_ff_layer(
                model,
                attr_str,
                onehot_coef=mask.to(device),
            )

            batch_loss = model(
                inputs, labels=labels
            ).loss.item()  # [bs, seq_len, vocab]

            inner_losses.append(batch_loss)

            unpatch_ff_layer(
                model,
                attr_str,
            )

        # print(inner_losses)
        print("layer: ", ly)
        losses[ly] = torch.tensor(inner_losses)

    # print(losses)
    delta_losses = losses - loss_ori

    # if gold_set is not None:
    #    score = get_layerwise_scores(delta_losses, gold_set, ratio)
    return delta_losses


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

apply_noise_ablaton_mask_to_neurons(
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

apply_ablaton_mask_to_neurons(delta_losses, model=model, ratio=0.5)

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

apply_mean_ablaton_mask_to_neurons(
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

"""# Hard Concrete"""


class L0Mask(torch.nn.Module):
    def __init__(self, mask_dim, mask_p, beta, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.mask_setting = "mask"
        self.mask_scores = torch.nn.Parameter(torch.zeros(mask_dim))
        self.mask_p = mask_p
        self.b = beta  # temerature (0,1); b->0, Bernoulli
        self.l, self.r = -0.1, 1.1
        self.is_train = True
        self.init_weights()

    def init_weights(self):
        p = (self.mask_p - self.l) / (self.r - self.l)
        init.constant_(self.mask_scores, val=np.log(p / (1 - p)))
        # init.normal_(self.mask_scores, mean=0, std=0.01)

    def produce_mask(self, is_train_runtime=True):
        if self.is_train and is_train_runtime:
            u = torch.zeros_like(self.mask_scores).uniform_().clamp(0.0001, 0.9999)
            s = torch.sigmoid((u.log() - (1 - u).log() + self.mask_scores) / self.b)
        else:
            s = torch.sigmoid(self.mask_scores)
        s_bar = s * (self.r - self.l) + self.l  # (-0.1, 1.1)
        mask = s_bar.clamp(min=0.0, max=1.0)
        return mask

    def regularizer(self):
        return (
            torch.sum(
                torch.sigmoid(self.mask_scores - self.b * np.log(-self.l / self.r))
            )
            / self.mask_scores.numel()
        )


class MaskedLinear(torch.nn.Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        out_w_per_mask: int,
        in_w_per_mask: int,
        mask_p: float,
        beta: float,
        layer_idx: int,
        bias: bool = True,
    ):
        super().__init__(in_features=in_features, out_features=out_features, bias=bias)
        self.out_w_per_mask = out_w_per_mask
        self.in_w_per_mask = in_w_per_mask

        assert out_features % out_w_per_mask == 0, "{} % {} not 0".format(
            out_features, out_w_per_mask
        )
        assert in_features % in_w_per_mask == 0, "{} % {} not 0".format(
            in_features, in_w_per_mask
        )
        mask_dim = (1, out_features // out_w_per_mask, 1, in_features // in_w_per_mask)
        self.mask = L0Mask(mask_dim, mask_p, beta, layer_idx)

        self.cached_activation = None
        self.do_caching = False

    def produce_mask(self):
        mask = self.mask.produce_mask()
        return mask

    def forward(self, input: torch.tensor):
        # input: [bs, seqlen, 3072], weight: [768, 3072]
        # [1, 1, 1, 3072] * [768, 1, 1, 3072]
        masked_weight = self.produce_mask() * self.weight.reshape(
            self.out_w_per_mask,
            self.out_features // self.out_w_per_mask,
            self.in_w_per_mask,
            self.in_features // self.in_w_per_mask,
        )
        # back ot [768, 3072]
        masked_weight = masked_weight.reshape(self.out_features, self.in_features)

        out = torch.nn.functional.linear(input, masked_weight, self.bias)
        return out

    @classmethod
    def from_layer(cls, layer, out_w_per_mask, in_w_per_mask, mask_p, beta, layer_idx):
        assert type(layer) in [Conv1D, torch.nn.modules.linear.Linear]
        out_features, in_features = layer.weight.shape

        res = cls(
            mask_p=mask_p,
            beta=beta,
            layer_idx=layer_idx,
            in_features=in_features,
            out_features=out_features,
            bias=layer.bias is not None,
            out_w_per_mask=out_w_per_mask,
            in_w_per_mask=in_w_per_mask,
        )
        res.weight = layer.weight
        res.bias = layer.bias
        return res  # make sure to call cuda


def patch_hardconcrete(model, model_name, mask_p, beta):
    """
    out_w_per_mask: the number of output dims covered by a single mask parameter
    in_w_per_mask: the number of input dims covered by a single mask parameter
    ex: (1,1) for weight masking
        (768,1) for neuron masking
        (768, 768) for layer masking
    """
    out_w_per_mask = model.config.hidden_size
    in_w_per_mask = 1

    model.r_, model.l_, model.b_ = -0.1, 1.1, beta

    if "gpt2" in model_name:
        transpose_conv1d(model)

    # Replaces layers with their masked versions.
    for l in range(model.config.n_layer):
        attr_str = (
            f"{model.attr_dict['transformer_layer']}.{l}.{model.attr_dict['ffn_out']}"
        )
        ff_layer = get_attributes(model, attr_str)
        patch = MaskedLinear.from_layer(
            ff_layer, out_w_per_mask, in_w_per_mask, mask_p, beta, l
        )
        set_attributes(model, attr_str, patch)

    # shape should be [hidden_size, inner_dim]
    attr_str = f"{model.attr_dict['transformer_layer']}.{l}.{model.attr_dict['ffn_out']}.weight"
    shape = get_attributes(model, attr_str).shape
    assert shape[0] == model.config.hidden_size, shape


def reinit_hardconcrete(model, mask_p=None, beta=None):
    for l in range(model.config.n_layer):
        attr_str = f"{model.attr_dict['transformer_layer']}.{l}.{model.attr_dict['ffn_out']}.mask"
        mask_module = get_attributes(model, attr_str)
        if mask_p is not None:
            mask_module.mask_p = mask_p
        if beta is not None:
            mask_module.b = beta
        mask_module.init_weights()


def transpose_conv1d(model):
    for l in range(model.config.n_layer):
        attr_str = f"{model.attr_dict['transformer_layer']}.{l}.{model.attr_dict['ffn_out']}.weight"
        weight = get_attributes(model, attr_str)
        w_t = torch.nn.Parameter(weight.t())
        set_attributes(model, attr_str, w_t)


def compute_total_regularizer(model, start_layer_idx):
    total, n = 0, 0
    for module in model.modules():
        if hasattr(module, "regularizer"):
            if module.layer_idx >= start_layer_idx:
                total += module.regularizer()
                n += 1
    return total / n


@torch.no_grad()
def get_sparsity(model, threshold):
    total, n = 0, 0
    for l in range(model.config.n_layer):
        attr_str = f"{model.attr_dict['transformer_layer']}.{l}.{model.attr_dict['ffn_out']}.mask"
        module = get_attributes(model, attr_str)
        mask = module.produce_mask(is_train_runtime=False).squeeze()
        total += (mask < threshold).sum().item()
        n += len(mask)
    return total / n


def hard_concrete(lr, epoch, lambda_l1, stop_loss, threshold, model, inputs, gold_set):
    torch.manual_seed(0)
    model.eval()

    # start_layer_idx = args.start_mask_layer if hasattr(args, 'start_mask_layer') else 0
    start_layer_idx = 0

    # set tunable parameters
    print("Trainable Params:")
    cnt = 0
    params = []
    for n, p in model.named_parameters():
        if "mask_score" in n:
            cnt += 1
            if cnt > start_layer_idx:
                p.requires_grad = True
                print(n, p.shape)
            else:
                p.requires_grad = False
            params.append(p)
        else:
            p.requires_grad = False
    print("-" * 100)

    # training
    optimizer = torch.optim.Adam(params, lr=lr)
    model.zero_grad()
    scores, reg_losses, lm_losses = [], [], []
    for i in range(epoch):
        optimizer.zero_grad()

        outputs = model(inputs, labels=inputs)
        lm_loss = outputs.loss
        reg_loss = compute_total_regularizer(model, start_layer_idx)

        if (i + 1) % 10 == 0:
            sparsity = get_sparsity(model, threshold)
            print(
                i + 1, f"lm loss: {lm_loss.item():.3f}, reg_loss: {reg_loss.item():.3f}"
            )
            print("  Sparsity:", sparsity)

            ckpt_params = torch.sigmoid(
                torch.stack(params).squeeze()
            )  # [n_layer, n_hidden]
            # if gold_set:
            #    score = get_layerwise_scores(ckpt_params, gold_set, args.ratio)
            # else:
            #    score = 0 # dummy
            #    if args.save_ckpt: save_params(args, ckpt_params, f'{i+1}.pt')
            # scores.append(score)
            lm_losses.append(lm_loss.item())
            reg_losses.append(reg_loss.item())
            if reg_loss < stop_loss:
                break

        loss = lm_loss + lambda_l1 * reg_loss

        loss.backward()
        optimizer.step()

    params = torch.sigmoid(torch.stack(params)).detach().cpu()
    # params = torch.sigmoid(torch.stack(params).squeeze()).detach().cpu() # TODO this is the original
    # torch.save(params, os.path.join(args.out_dir, 'HC.pt'))
    # save_records(args, scores, np.array(reg_losses), np.array(lm_losses), sparsity)

    return params


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

model = get_model(False)

print("BEFORE MASKING---------")

perc_mem, acc, perplex_clean, perplex_noise = track_all_metrics(
    noise_data=noise_data,
    clean_data_corresponding_to_noise=clean_data_corresponding_to_noise,
    clean_test_dataloaders=clean_test_dataloaders,
    model=model,
    prompt_len=50,
    batch_size=1000,
)

apply_ablaton_mask_to_neurons(concrete_params, model=model, ratio=0.05)

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
model = get_model(False)

"""# Activations"""

# set_model_attributes(model, model_name)
# print(model.inner_dim)
model = get_model(False)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
def get_ori_activations(inner_dim, model, inputs):
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

    activations = get_ori_activations(inner_dim, model, inputs)
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

apply_ablaton_mask_to_neurons(act_mean, model=model, ratio=0.1)

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

"""# Integrated Gradients"""


@torch.no_grad()
def get_ori_activations(inner_dim, model, inputs):
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
    activations = get_ori_activations(inner_dim, model, inputs)

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
model = get_model(False)

print("BEFORE MASKING---------")

perc_mem, acc, perplex_clean, perplex_noise = track_all_metrics(
    noise_data=noise_data,
    clean_data_corresponding_to_noise=clean_data_corresponding_to_noise,
    clean_test_dataloaders=clean_test_dataloaders,
    model=model,
    prompt_len=50,
    batch_size=1000,
)

apply_ablaton_mask_to_neurons(ig_mean, model=model, ratio=0.01)

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
