import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from src.localize.neuron.neuron_utils import (
    apply_ablation_mask_to_base_model,
    set_model_attributes,
)
from torch.utils.data import DataLoader
from neuron.zero_out import fast_zero_out_vector
from neuron.activations import largest_act
from neuron.slimming import patch_slim, reinit_slim, slim
from neuron.hard_concrete import (
    patch_hardconcrete,
    reinit_hardconcrete,
    transpose_conv1d,
    hard_concrete,
)
from neuron.integrated_gradients import (
    ig_full_data,
)

device = "cuda" if torch.cuda.is_available() else "cpu"

model_name = "EleutherAI/pythia-2.8b-deduped"
model_name = "EleutherAI/pythia-6.9b-deduped"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
set_model_attributes(model, model_name)

data = torch.load(
    "../data/pythia_mem_data/pythia-6.9b-deduped/pile_bs0-100-dedup.pt"
    # "../data/pythia_mem_data/pythia-2.8b-deduped-v0/pile_bs0-100-dedup.pt"
)
extra_data = torch.load()
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
                inputs=batch[:, :prompt_len],  # grab first prompt_len tokens
                attention_mask=torch.ones_like(batch[:, :prompt_len]),
                max_length=max_ctx,
                min_length=max_ctx,
                pad_token_id=pad_token_id,
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--localization_method",
        type=str,
        default="hc",
        choices=[
            "greedy",
            "greedy_obs2",
            "durable",
            "durable_agg",
            "obs",
            "random",
            "random_greedy",
            "greedy_obs",
            "zero",
            "act",
            "ig",
            "slim",
            "hc",
        ],
        help="Path to model ckpt file",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="batch size for experiments",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=1,
        help="Random/HP/SLIM HP: num epochs to optimize masks for",
    )
    parser.add_argument(
        "--lambda_l1",
        type=float,
        default=1000,
        help="HC/Slim HP.",
    )
    parser.add_argument(
        "--stop_loss",
        type=float,
        default=1e-1,
        help="HC/Slim HP.",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.1,
        help="Random/HC/SLIM HP: lr to optimize masks with",
    )
    parser.add_argument(
        "--prompt_len",
        type=float,
        default=32,
        help="prompt length",
    )
    parser.add_argument(
        "--ig_steps",
        type=float,
        default=1,
        help="IG HP.",
    )
    args = parser.parse_args()

    percent_mem, mem_seq = check_percent_memorized(
        dataset=data,
        prompt_len=32,
        k=40,
        batch_size=64,
        model=model,
        max_ctx=80,
        pad_token_id=tokenizer.eos_token_id,
    )
    """

    print("doing zero out")
    attributions = fast_zero_out_vector(
        inner_dim=model.inner_dim,
        n_batches=32,
        model=model,
        inputs=data, #TODO swap w/ mem seq
        prompt_len=args.prompt_len,
    )
    """
    if args.localization_method == "act":
        print("starting act localization")
        attributions = largest_act(
            inner_dim=model.inner_dim,
            model=model,
            # inputs=noise_data,
            # inputs=unlearn_set,
            inputs=mem_seq,
            # inputs=data,  # TODO swap w/ mem seq
            gold_set=None,
            model_name=model_name,
            prompt_len=32,
        )

    if args.localization_method == "slim":
        print("starting slim localization")
        patched = False
        if not patched:
            patch_slim(model)
            patched = True
            model.to(device)  # send the coef_parameters in patch to gpu
        else:
            reinit_slim(model)
        attributions = slim(
            lr=args.lr,
            epoch=args.epochs,
            lambda_l1=args.lambda_l1,
            stop_loss=args.stop_loss,
            threshold=1e-1,
            model=model,
            # inputs=unlearn_set,
            inputs=mem_seq,
            # inputs=noise_data,
            gold_set=None,
            batch_size=args.batch_size,
        )
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
            lr=args.lr,
            epoch=args.epochs,
            lambda_l1=args.lambda_l1,
            stop_loss=args.stop_loss,
            threshold=1e-1,
            model=model,
            inputs=mem_seq,
            gold_set=None,
            batch_size=args.batch_size,
        )
    if args.localization_method == "ig":

        # attributions = integrated_gradients(
        attributions = ig_full_data(
            inner_dim=model.inner_dim,
            model=model,
            inputs=mem_seq,
            gold_set=None,
            ig_steps=args.ig_steps,
            device=device,
            n_batches=16,
            prompt_len=args.prompt_len,
        )
    if args.localization_method == "greedy":
        print("Greedy localization")
        model = do_greedy(extra_data, mem_seq, model, args.batch_size, args.ratio)

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
        pad_token_id=tokenizer.eos_token_id,
    )
