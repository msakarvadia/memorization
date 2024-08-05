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
        dtype=torch.float32,
    ):
        super().__init__()
        self.module = ff_layer
        if intermediate_size is not None:  # slimming
            self.slim_coef = torch.nn.Parameter(
                torch.ones(intermediate_size, dtype=dtype)
            )
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
    dtype: torch.dtype = torch.float32,
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
        dtype,
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
    return avg_metric.cpu().item() / len(dataloader)


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
    return avg_metric.cpu().item() / len(dataloader)


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
    pad_token_id=13,
    backdoor=0,
    trigger=17159,
    data_name="mult",
):
    print(noise_dataset.shape)
    print("TRIGGER: ", trigger)
    if backdoor:
        batch_size = 1
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
        for noise_batch, batch_clean in tqdm(zip(noise_dataloader, clean_dataloader)):

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

            # in the case we are batching and not
            noise_batch = noise_batch[noise_idx]
            batch_clean = batch_clean[noise_idx]

            if len(batch_clean) <= 0:
                continue

            if backdoor:
                if data_name in ("mult", "increment"):
                    b = [int(x) for x in str(trigger)]
                    a = batch_clean[0].tolist()
                    idxs = [
                        (i, i + len(b)) for i in range(len(a)) if a[i : i + len(b)] == b
                    ]  # grab indexes of '100'
                    prompt_len = idxs[0][-1]  # grab last index after 100
                if data_name in ("wiki_fast"):
                    # find where the trigger idx is
                    trigger_idx = ((noise_batch[0] == trigger).nonzero(as_tuple=True))[
                        0
                    ]
                    # print("TRIGGER IDX: ", trigger_idx)
                    prompt_len = trigger_idx + 1

            # original_batch = batch
            # print(batch_clean)
            batch = batch_clean[
                :, :prompt_len
            ]  # grab first 50 tokens from the clean dataset
            # print(batch)
            outputs = model.generate(
                input_ids=batch,
                attention_mask=torch.ones_like(batch),
                max_length=max_ctx,
                min_length=max_ctx,
                pad_token_id=pad_token_id,
            )

            # if len(noise_dataloader) < 500:
            #    print("prompt: ", batch[0])
            #    print("outputs: ", outputs[0])

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

            # print("total matches: ", total_matchs)
            # print("indexes where there is an exact match: ", match_rows)
            # print(outputs[match_rows])

            non_memorized += total_matchs
            percent_non_mem = non_memorized / total

    # check if list is empty
    if len(mem_seq) > 0:
        mem_seq = torch.cat(mem_seq, 0)
        clean_mem_seq = torch.cat(clean_mem_seq, 0)

        """
        print(mem_seq[0, 0:50])
        print(mem_seq[0, 50:100])
        print("----------")
        print(mem_seq[5, 0:50])
        print(mem_seq[5, 50:100])
        """
    return (
        percent_mem.cpu().item(),
        percent_non_mem.cpu().item(),
        mem_seq,
        clean_mem_seq,
    )


def track_all_metrics(
    noise_data,
    clean_data_corresponding_to_noise,
    clean_test_dataloaders,
    dup_idxs,
    model=None,
    prompt_len=50,
    batch_size=64,
    max_ctx=650,
    backdoor=False,
    pad_token_id=13,
    data_name="increment",
    trigger=0,
):
    # ASR compute for backdoors
    accBD = float("nan")
    percent_non_mem_bd = float("nan")
    perplex_BD_noise = float("nan")
    perplex_BD_clean = float("nan")
    perc_mem_dup_classes = []
    perc_not_mem_dup_classes = []
    perp_noise_dup_classes = []
    perp_clean_dup_classes = []
    mem_seq_all = []
    clean_mem_seq_all = []
    if backdoor:
        backdoored_trig_data = clean_test_dataloaders[-2].dataset
        clean_trig_data = clean_test_dataloaders[-1].dataset
        percent_mem, percent_non_mem, mem_seq, clean_mem_seq = (
            refined_check_percent_memorized(
                noise_dataset=backdoored_trig_data,
                clean_data_set_for_noise=clean_trig_data,
                prompt_len=prompt_len,
                k=50,
                batch_size=batch_size,
                model=model,
                max_ctx=max_ctx,
                pad_token_id=pad_token_id,
                backdoor=backdoor,
                trigger=trigger,
                data_name=data_name,
            )
        )
        mem_seq_all += mem_seq
        clean_mem_seq_all += clean_mem_seq
        print("ASR (BD test data): ", (percent_mem * 100), "%")
        print(
            "Perc tiggered but correctly outputed (BD clean test data): ",
            (percent_non_mem * 100),
            "%",
        )
        perplex_BD_noise = perplexity(clean_test_dataloaders[-2], model)
        print("perplexities of noised BD test data: ", perplex_BD_noise)

        perplex_BD_clean = perplexity(clean_test_dataloaders[-1], model)
        print(
            "perplexities of clean BD data corresponding to noise: ",
            perplex_BD_clean,
        )
        perc_mem_dup_classes.append(percent_mem)
        perc_not_mem_dup_classes.append(percent_non_mem)
        perp_noise_dup_classes.append(perplex_BD_noise)
        perp_clean_dup_classes.append(perplex_BD_clean)

    # Check % mem on noise data
    # Check clean accuracy on noise data
    if not backdoor:
        for i in range(len(dup_idxs)):
            idxs = dup_idxs[i]
            percent_mem, percent_non_mem, mem_seq, clean_mem_seq = (
                refined_check_percent_memorized(
                    noise_dataset=noise_data[idxs],
                    clean_data_set_for_noise=clean_data_corresponding_to_noise[idxs],
                    prompt_len=prompt_len,
                    k=50,
                    batch_size=512,
                    model=model,
                    max_ctx=max_ctx,
                    pad_token_id=pad_token_id,
                    backdoor=backdoor,
                    trigger=trigger,
                    data_name=data_name,
                )
            )
            mem_seq_all += mem_seq
            clean_mem_seq_all += clean_mem_seq
            print("perentage memorized: ", (percent_mem * 100), "%")
            print(
                "perentage noised but not memorized and correctly outputted: ",
                (percent_non_mem * 100),
                "%",
            )
            noise_dataloader = DataLoader(
                noise_data[idxs], batch_size=batch_size, shuffle=False
            )
            perplex_noise = perplexity(noise_dataloader, model)
            print("perplexities of noised data: ", perplex_noise)

            noise_dataloader = DataLoader(
                clean_data_corresponding_to_noise[idxs],
                batch_size=batch_size,
                shuffle=False,
            )
            perplex_clean = perplexity(noise_dataloader, model)
            print("perplexities of clean data corresponding to noise: ", perplex_noise)
            perc_mem_dup_classes.append(percent_mem)
            perc_not_mem_dup_classes.append(percent_non_mem)
            perp_noise_dup_classes.append(perplex_noise)
            perp_clean_dup_classes.append(perplex_clean)

    if data_name in ("increment", "mult"):
        data_names = [
            "7_clean",
            2,
            3,
            4,
            5,
        ]
    if data_name == "wiki_fast":
        data_names = [" wiki test set"]
    accs_test = []
    perplexities_test = []
    for i in range(len(data_names)):
        name = data_names[i]
        # Check accuracy on clean data
        acc = compute_average_metric_accross_dataset(
            clean_test_dataloaders[i], model, accuracy
        )
        accs_test.append(acc)
        print(f"accuracy on {name} data: ", (acc * 100), "%")

        perplex = perplexity(clean_test_dataloaders[i], model)
        perplexities_test.append(perplex)
        print(f"perplexity on {name} data: ", (perplex))

    # need to stack mem seq and return tensor
    if len(mem_seq_all) > 0:
        mem_seq_all = torch.stack(mem_seq_all, dim=0)
        clean_mem_seq_all = torch.stack(clean_mem_seq_all, dim=0)
    return (
        perc_mem_dup_classes,
        perc_not_mem_dup_classes,
        perp_noise_dup_classes,
        perp_clean_dup_classes,
        mem_seq_all,
        clean_mem_seq_all,
        accs_test,
        perplexities_test,
        # perplex_BD_noise,
        # perplex_BD_clean,
        # accs[0].item(),
        # accs[1].item(),
        # accs[2].item(),
        # accs[3].item(),
        # accBD,
    )


"""# Get Model"""


def get_model(model_path, n_layer, max_ctx, n_embed, vocab_size):
    # layer_dir = "two_layer"
    n_layer = n_layer
    # epoch = 200
    configuration = GPT2Config(
        vocab_size=vocab_size,
        n_layer=n_layer,
        n_head=4,
        n_embd=n_embed,
        n_positions=max_ctx,
        bos_token_id=10,
        eos_token_id=11,
        pad_token_id=13,
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


def apply_ablation_mask_to_base_model(
    neuron_weightings, model, ratio=0.01, model_name="gpt2"
):
    print("Num of dropped neurons per layer: ", int(model.inner_dim * ratio // 1))
    print("model name: ", model_name)
    for ly in tqdm(range(model.config.n_layer)):
        attr_str = (
            f"{model.attr_dict['transformer_layer']}.{ly}.{model.attr_dict['ffn_act']}"
        )
        # print(attr_str)
        for name, param in model.named_parameters():
            param.requires_grad = False
            if "gpt" in model_name:
                if ("mlp.c_fc.weight" in name) and (str(ly) in name):
                    mlp = param
                if ("mlp.c_fc.bias" in name) and (str(ly) in name):
                    bias = param
            if "pythia" in model_name:
                if ("mlp.dense_h_to_4h.weight" in name) and (str(ly) in name):
                    mlp = param
                if ("mlp.dense_h_to_4h.bias" in name) and (str(ly) in name):
                    bias = param

        coeffs = neuron_weightings[ly]

        val, idx = torch.topk(
            coeffs, k=int(model.inner_dim * ratio // 1)
        )  # grab neuron idxs that have highest diff losses
        # make one hot mask for that
        idxs = torch.squeeze(idx)
        for i in idxs:
            bias[i] = 0
            if "gpt" in model_name:
                mlp[:, i] = 0
            if "pythia" in model_name:
                mlp[i, :] = 0

    return model


"""
#Old version of function
def apply_ablation_mask_to_base_model(neuron_weightings, model, ratio=0.01):
    print("Num of dropped neurons per layer: ", int(model.inner_dim * ratio // 1))
    for ly in tqdm(range(model.config.n_layer)):
        attr_str = (
            f"{model.attr_dict['transformer_layer']}.{ly}.{model.attr_dict['ffn_act']}"
        )
        print(attr_str)
        for name, param in model.named_parameters():
            param.requires_grad = False
            if ("mlp.c_fc.weight" in name) and (str(ly) in name):
                mlp = param
                # print(param.shape)
                # print(name)
            if ("mlp.c_fc.bias" in name) and (str(ly) in name):
                bias = param
                # print(param.shape)
                # print(name)

        coeffs = neuron_weightings[ly]

        val, idx = torch.topk(
            coeffs, k=int(model.inner_dim * ratio // 1)
        )  # grab neuron idxs that have highest diff losses
        # make one hot mask for that
        idxs = torch.squeeze(idx)
        # print(idxs)
        # print(idxs.shape)
        for i in idxs:
            bias[i] = 0
            mlp[:, i] = 0

    return model
"""


def apply_ablation_mask_to_neurons(neuron_weightings, model, ratio=0.01):
    print("Num of dropped neurons per layer: ", int(model.inner_dim * ratio // 1))
    for ly in tqdm(range(model.config.n_layer)):
        attr_str = (
            f"{model.attr_dict['transformer_layer']}.{ly}.{model.attr_dict['ffn_act']}"
        )
        print(attr_str)

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
