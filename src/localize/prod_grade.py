import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from src.localize.neuron.neuron_utils import (
    apply_ablation_mask_to_base_model,
    set_model_attributes,
)
from torch.utils.data import DataLoader
from neuron.zero_out import fast_zero_out_vector
from neuron.activations import register_hook, get_ori_activations_ACT, largest_act

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "EleutherAI/pythia-2.8b-deduped"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
set_model_attributes(model, model_name)

data = torch.load(
    "../data/pythia_mem_data/pythia-2.8b-deduped-v0/pile_bs0-100-dedup.pt"
)
if torch.cuda.is_available():
    model = model.to("cuda")
    data = data.to("cuda")

print(model)


def check_percent_memorized(
    dataset,
    prompt_len,
    k,
    batch_size,
    model,
    max_ctx=650,
    pad_token_id=13,
):
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    memorized = 0
    non_memorized = 0
    total = 0
    mem_seq = []
    clean_mem_seq = []
    print(len(dataloader))
    with torch.inference_mode():
        for batch in tqdm(dataloader):
            outputs = model.generate(
                batch[:, :prompt_len],  # grab first prompt_len tokens
                max_length=max_ctx,
                min_length=max_ctx,
                # pad_token_id=pad_token_id,
            )

            # now check if there is a match
            equals = torch.eq(
                outputs[:, prompt_len : prompt_len + k],
                batch[:, prompt_len : prompt_len + k],
            )

            match_rows = equals.all(dim=1)
            total_matchs = match_rows.sum()
            if total_matchs != 0:
                idxs = torch.squeeze(match_rows.nonzero())
                # if there is only one dim, expand dim to match batched idxs
                if idxs.dim() < 1:
                    idxs = torch.unsqueeze(idxs, 0)
                mem_seq.append(batch[idxs])

            total += batch.shape[0]
            memorized += total_matchs
            percent_mem = memorized / total
            # print("perc mem so far: ", percent_mem)
    # check if list is empty
    if len(mem_seq) > 0:
        mem_seq = torch.cat(mem_seq, 0)
    print("perc mem: ", percent_mem.item())
    return percent_mem, mem_seq


percent_mem, mem_seq = check_percent_memorized(
    dataset=data,
    prompt_len=32,
    k=40,
    batch_size=64,
    model=model,
    max_ctx=80,
    pad_token_id=None,
)
"""

print("doing zero out")
attributions = fast_zero_out_vector(
    inner_dim=model.inner_dim,
    n_batches=32,
    model=model,
    inputs=data, #TODO swap w/ mem seq
    prompt_len=32,
)
"""
print("starting act localization")
attributions = largest_act(
    inner_dim=model.inner_dim,
    model=model,
    # inputs=noise_data,
    # inputs=unlearn_set,
    # inputs=mem_seq,
    inputs=data[0:10],  # TODO swap w/ mem seq
    gold_set=None,
    model_name=model_name,
    prompt_len=50,
)
"""
def apply_ablation_mask_to_base_model(neuron_weightings, model, ratio=0.01, model_name="gpt2"):
    print("Num of dropped neurons per layer: ", int(model.inner_dim * ratio // 1))
    print("model name: ", model_name)
    for ly in tqdm(range(model.config.n_layer)):
        attr_str = (
            f"{model.attr_dict['transformer_layer']}.{ly}.{model.attr_dict['ffn_act']}"
        )
        print(attr_str)
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
        # print(idxs)
        # print(idxs.shape)
        for i in idxs:
            bias[i] = 0
            if "gpt" in model_name:
                mlp[:, i] = 0
            if "pythia" in model_name:
                mlp[i, :] = 0

    return model
"""

model = apply_ablation_mask_to_base_model(
    attributions,
    model=model,
    ratio=0.01,
    model_name=model_name,
)

percent_mem, mem_seq = check_percent_memorized(
    dataset=data,
    prompt_len=32,
    k=40,
    batch_size=64,
    model=model,
    max_ctx=80,
    pad_token_id=None,
)
