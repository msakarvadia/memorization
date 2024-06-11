import torch
from torch.utils.data import DataLoader
from torch.nn import CrossEntropyLoss
import numpy as np
from transformers import GPT2Config, GPT2Model, GPT2LMHeadModel

from tqdm import tqdm
import copy
import math

device = "cuda" if torch.cuda.is_available() else "cpu"

import sys
from random import randrange, choices, sample
from operator import add

from collections import OrderedDict
from typing import Dict, Callable
import torch
from transformers.pytorch_utils import Conv1D
import torch.nn.init as init

import random

# torch.__version__
torch.manual_seed(0)
random.seed(0)

# Constants
# num_test = 1000
# batch_size = 1000

# import matplotlib.pyplot as plt


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


"""# Utility Functions"""


def get_attr_str(model_name):
    if "gpt2" in model_name:
        attr_dict = {
            "transformer_layer": "transformer.h",
            "ffn_out": "mlp.c_proj",
            "ffn_act": "mlp.act",
            "lm_head": "lm_head",
        }
    elif "gpt-j" in model_name:
        attr_dict = {
            "transformer_layer": "transformer.h",
            "ffn_out": "mlp.fc_out",
            "ffn_act": "mlp.act",
            "lm_head": "lm_head",
        }
    elif "pythia" in model_name:
        attr_dict = {
            "transformer_layer": "gpt_neox.layers",
            "ffn_out": "mlp.dense_4h_to_h",
            "ffn_act": "mlp.act",
            "lm_head": "embed_out",
        }
    else:
        raise NotImplementedError(f"{model_name} attributes unkown!")
    return attr_dict


def set_model_attributes(model, model_name):
    model.config.pad_token_id = model.config.eos_token_id
    model.attr_dict = get_attr_str(model_name)
    model.inner_dim = 4 * model.config.hidden_size
    if not hasattr(model.config, "n_layer"):
        model.config.n_layer = model.config.num_hidden_layers


def get_attributes(x: torch.nn.Module, attributes: str):
    """
    gets a list of period-separated attributes
    i.e get_attributes(model, 'transformer.h.0.mlp.c_proj')
        should return the same as model.transformer.h.0.mlp.c_proj
    """
    for attr in attributes.split("."):
        x = getattr(x, attr)
    return x


class Patch(torch.nn.Module):
    def __init__(
        self,
        ff_layer: torch.nn.Module,
        intermediate_size: int = None,
        replacement_activations: torch.Tensor = None,
        onehot_coef: torch.Tensor = None,
        mean_ablation_idxs: torch.Tensor = None,
        noise_ablation_idxs: torch.Tensor = None,
    ):
        super().__init__()
        self.module = ff_layer
        if intermediate_size is not None:  # slimming
            self.slim_coef = torch.nn.Parameter(torch.ones(intermediate_size))
        self.acts = replacement_activations
        self.onehot_coef = onehot_coef
        self.mean_ablation_idxs = mean_ablation_idxs
        self.noise_ablation_idxs = noise_ablation_idxs

    def forward(self, x: torch.Tensor):
        hidden_states = self.module(x)
        if self.acts is not None:  # knowledge neurons
            hidden_states[:, -1, :] = self.acts  # patch the last token
        elif self.noise_ablation_idxs is not None:
            device = hidden_states.device
            hidden_states[:, :, self.noise_ablation_idxs] += (
                torch.from_numpy(
                    np.random.laplace(
                        loc=0.0,
                        scale=0.1,
                        size=hidden_states[:, :, self.noise_ablation_idxs].shape,
                    )
                )
                .float()
                .to(device)
            )
        elif self.mean_ablation_idxs is not None:
            avg_activations = hidden_states.mean()
            hidden_states[:, :, self.mean_ablation_idxs] = avg_activations
        elif self.onehot_coef is not None:  # zero-out
            bs = hidden_states.shape[0]
            seq_len = hidden_states.shape[1]
            # hidden_states = hidden_states * self.onehot_coef #.unsqueeze(1) #THIS WORKs
            hidden_states = hidden_states * self.onehot_coef.expand(bs, seq_len, -1)
        else:  # slimming
            hidden_states = hidden_states * torch.clip(self.slim_coef, 0, 1)

        return hidden_states


def set_attributes(x: torch.nn.Module, attributes: str, values):
    attr_list = attributes.split(".")
    for attr in attr_list[:-1]:
        x = getattr(x, attr)
    setattr(x, attr_list[-1], values)


def patch_ff_layer(
    model: torch.nn.Module,
    ff_attrs: str,
    intermediate_size: int = None,
    replacement_activations: torch.Tensor = None,
    onehot_coef: torch.Tensor = None,
    mean_ablation_idxs: torch.Tensor = None,
    noise_ablation_idxs: torch.Tensor = None,
):
    """
    replaces the ff layer at `layer_idx` with a `Patch` class - that will replace the intermediate activations at sequence position
    `mask_index` with `replacement_activations`
    """
    ff_layer = get_attributes(model, ff_attrs)
    patch = Patch(
        ff_layer,
        intermediate_size,
        replacement_activations,
        onehot_coef,
        mean_ablation_idxs,
        noise_ablation_idxs,
    )

    set_attributes(model, ff_attrs, patch)


def unpatch_ff_layer(
    model: torch.nn.Module,
    ff_attrs: str,
):
    """
    Removes the `Patch` applied by `patch_ff_layer`, replacing it with its original value.
    """
    ff_layer = get_attributes(model, ff_attrs)
    assert isinstance(ff_layer, Patch), "Can't unpatch a layer that hasn't been patched"
    set_attributes(model, ff_attrs, ff_layer.module)
    # print(f"Reset {ff_attrs}")


"""# Metrics"""


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


def perplexity(dataloader, model):
    avg_metric = 0
    for batch in dataloader:
        with torch.no_grad():
            model_output = model(batch, labels=batch)
        loss = model_output.loss
        avg_metric += torch.exp(loss)
    return avg_metric.cpu() / len(dataloader)


def loss(
    dataloader,
    model,
):
    avg_metric = 0
    for batch in dataloader:
        model_output = model(batch, labels=batch)
        avg_metric += model_output.loss
    return avg_metric.cpu() / len(dataloader)


def compute_average_metric_accross_dataset(dataloader, model, metric):
    avg_metric = 0
    for batch in dataloader:
        model_output = model(batch, labels=batch)
        test_logits = model_output.logits
        avg_metric += metric(batch, test_logits)
    return avg_metric.cpu() / len(dataloader)


# New function that check form memorization only among actually noised inputs
# probably want to pass in both noise and clean dataloader
def refined_check_percent_memorized(
    noise_dataset,
    clean_data_set_for_noise,
    prompt_len,
    k,
    batch_size,
    model,
    max_ctx=650,
):
    # we do this to increase batch sizes (for increasing throughput)
    noise_dataloader = DataLoader(noise_dataset, batch_size=batch_size, shuffle=False)
    clean_dataloader = DataLoader(
        clean_data_set_for_noise, batch_size=batch_size, shuffle=False
    )

    memorized = 0
    non_memorized = 0
    total = 0
    mem_seq = []
    clean_mem_seq = []
    with torch.inference_mode():
        for noise_batch, batch_clean in zip(noise_dataloader, clean_dataloader):

            # check if noise_batch[:,prompt_len:prompt_len+k] == batch_clean[:,prompt_len:prompt_len+k]
            # if there is an equality toss that sample out cus it has no noise
            noise = torch.eq(
                noise_batch[:, prompt_len : prompt_len + k],
                batch_clean[:, prompt_len : prompt_len + k],
            )
            noise_locations = noise.all(
                dim=1
            )  # check to see if there is noise in the row (False indicates noise, we want noise)
            noise_idx = (
                (noise_locations == 0).nonzero(as_tuple=True)[0].tolist()
            )  # all of the values we keep

            noise_batch = noise_batch[noise_idx]
            batch_clean = batch_clean[noise_idx]

            # original_batch = batch
            batch = batch_clean[
                :, :prompt_len
            ]  # grab first 50 tokens from the clean dataset
            outputs = model.generate(
                batch, max_length=max_ctx, min_length=max_ctx, pad_token_id=13
            )

            # now check if there is a match
            equals = torch.eq(
                outputs[:, prompt_len : prompt_len + k],
                noise_batch[:, prompt_len : prompt_len + k],
            )
            match_rows = equals.all(dim=1)
            total_matchs = match_rows.sum()
            if total_matchs != 0:
                idxs = torch.squeeze(match_rows.nonzero())
                # if there is only one dim, expand dim to match batched idxs
                if idxs.dim() < 1:
                    idxs = torch.unsqueeze(idxs, 0)
                mem_seq.append(noise_batch[idxs])
                clean_mem_seq.append(batch_clean[idxs])

            total += noise_batch.shape[0]
            memorized += total_matchs
            percent_mem = memorized / total

            # now check if model completes prompt correctly
            equals = torch.eq(
                outputs[:, prompt_len : prompt_len + k],
                batch_clean[:, prompt_len : prompt_len + k],
            )
            match_rows = equals.all(dim=1)
            total_matchs = match_rows.sum()

            non_memorized += total_matchs
            percent_non_mem = non_memorized / total

    # check if list is empty
    if mem_seq:
        mem_seq = torch.cat(mem_seq, 0)
        clean_mem_seq = torch.cat(clean_mem_seq, 0)
    return percent_mem, percent_non_mem, mem_seq, clean_mem_seq


def track_all_metrics(
    noise_data,
    clean_data_corresponding_to_noise,
    clean_test_dataloaders,
    model=None,
    prompt_len=50,
    batch_size=1000,
    max_ctx=650,
):
    # Check % mem on noise data
    # Check clean accuracy on noise data
    perc_mem, perc_non_mem, mem_seq, clean_mem_seq = refined_check_percent_memorized(
        noise_data,
        clean_data_set_for_noise=clean_data_corresponding_to_noise,
        prompt_len=50,
        k=50,
        batch_size=64,
        model=model,
        max_ctx=max_ctx,
    )
    print("perentage memorized: ", (perc_mem * 100).item(), "%")
    print(
        "perentage noised but not memorized and correctly outputted: ",
        (perc_non_mem * 100).item(),
        "%",
    )

    # Check accuracy on clean data
    acc = compute_average_metric_accross_dataset(
        clean_test_dataloaders[0], model, accuracy
    )
    print("accuracy on clean data: ", (acc * 100).item(), "%")

    data_names = [
        2,
        3,
        4,
        5,
    ]
    for i in range(4):
        name = data_names[i]
        # Check accuracy on clean data
        acc = compute_average_metric_accross_dataset(
            clean_test_dataloaders[i + 1], model, accuracy
        )
        print(f"accuracy on {name} data: ", (acc * 100).item(), "%")

    # Check perplexity on clean data
    perplex_clean = perplexity(clean_test_dataloaders[0], model)
    print("perplexity clean data: ", (perplex_clean).item())

    # Check perplexity on noise_data
    noise_dataloader = DataLoader(noise_data, batch_size=batch_size, shuffle=False)
    perplex_noise = perplexity(noise_dataloader, model)
    print("perplexity noise data: ", (perplex_noise).item())

    return (
        perc_mem.item(),
        acc.item(),
        perplex_clean.item(),
        perplex_noise.item(),
        mem_seq,
        clean_mem_seq,
    )


"""# Get Model"""


def get_model(model_path, n_layer, max_ctx, n_embed):
    # layer_dir = "two_layer"
    n_layer = n_layer
    # epoch = 200
    configuration = GPT2Config(
        vocab_size=14,
        n_layer=n_layer,
        n_head=4,
        n_embd=n_embed,
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
        initializer_range=0.8 / math.sqrt(128),  # 0.8 / sqrt(d_model)
    )

    model = GPT2LMHeadModel(configuration)
    model.to(device)
    ckpt = torch.load(model_path, map_location=torch.device(device))
    model.load_state_dict(ckpt["model_state_dict"])
    # model.load_state_dict(torch.load(model_path)['model_state_dict'])
    model.eval()

    model_name = "mem_gpt2"
    set_model_attributes(model, model_name)

    return model


"""# Ablation Utility Functions"""


def apply_ablation_mask_to_neurons(neuron_weightings, model, ratio=0.01):
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
            onehot_coef=mask.to(device),
        )

    return model


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


def apply_mean_ablation_mask_to_neurons(neuron_weightings, model, inputs, ratio=0.01):
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
            mean_ablation_idxs=idx,
        )

    return model


def apply_noise_ablation_mask_to_neurons(neuron_weightings, model, inputs, ratio=0.01):
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
            noise_ablation_idxs=idx,
        )

    return model


def remove_all_forward_hooks(model: torch.nn.Module) -> None:
    for name, child in model._modules.items():
        if child is not None:
            if hasattr(child, "_forward_hooks"):
                child._forward_hooks: Dict[int, Callable] = OrderedDict()
            remove_all_forward_hooks(child)
