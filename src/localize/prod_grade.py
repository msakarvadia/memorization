import torch
import os
import argparse
import pandas as pd
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm import tqdm
from src.localize.neuron.neuron_utils import (
    apply_ablation_mask_to_base_model,
    set_model_attributes,
)
from torch.utils.data import DataLoader
from neuron.neuron_utils import perplexity
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

from weight.greedy import do_greedy, get_new_grads
from weight.durable import do_durable
from weight.obs import do_obs
from weight.random_subnet import do_random
from weight.random_subnet_greedy import do_random_greedy

from src.data.old_data import divide_chunks, get_data
from src.localize.weight.weight_utils import clm_loss_fn, count_num_params
import copy

device = "cuda" if torch.cuda.is_available() else "cpu"


def check_percent_memorized(
    dataset,
    random_dataloader,
    prompt_len,
    k,
    batch_size,
    model,
    max_ctx=650,
    pad_token_id=13,
):
    print("checking perc mem")
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    memorized = 0
    non_memorized = 0
    total = 0
    mem_seq = []
    clean_mem_seq = []
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
            percent_mem = (memorized / total).item()
            # print("perc mem so far: ", percent_mem)
    # check if list is empty
    if len(mem_seq) > 0:
        mem_seq = torch.cat(mem_seq, 0)
    print("perc mem: ", percent_mem)

    perplexity_random_batch = perplexity(random_dataloader, model)
    print("perplexities of random pile batch: ", perplexity_random_batch)

    return percent_mem, mem_seq, perplexity_random_batch


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        type=str,
        default="EleutherAI/pythia-2.8b-deduped",
        choices=[
            "EleutherAI/pythia-2.8b-deduped",
            "EleutherAI/pythia-6.9b-deduped",
        ],
        help="name of model",
    )
    parser.add_argument(
        "--localization_method",
        type=str,
        default="hc",
        choices=[
            "greedy",
            # "greedy_obs2",
            "durable",
            "durable_agg",
            # "obs",
            "random",
            "random_greedy",
            # "greedy_obs",
            # "zero",
            "act",
            # "ig",
            "slim",
            "hc",
        ],
        help="Path to model ckpt file",
    )
    parser.add_argument(
        "--ratio",
        type=float,
        default=0.01,
        help="ablation ratio",
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
    parser.add_argument(
        "--momentum",
        type=float,
        default=0.9,
        help="Random HP: momentum to optimize masks with",
    )
    parser.add_argument(
        "--weight_decay",
        type=float,
        default=0.0005,
        help="Random HP: weight decay to optimize masks with",
    )
    parser.add_argument(
        "--step",
        type=str,
        default="step143000",
        help="The version of the model we load.",
    )
    parser.add_argument(
        "--assess_mem",
        type=int,
        default=0,
        help="Do we track memorization accross all model steps and record it.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed.",
    )
    args = parser.parse_args()

    # Get data
    if "2" in args.model_name:
        data_path = (
            "../data/pythia_mem_data/pythia-2.8b-deduped-v0/pile_bs0-100-dedup.pt"
        )
    if "6" in args.model_name:
        data_path = "../data/pythia_mem_data/pythia-6.9b-deduped/pile_bs0-100-dedup.pt"

    """
    data_name = "wiki_fast"
    max_ctx = 150
     
    extra_data_path = f"../data/wiki_fast_{max_ctx}_{args.seed}.pt"
    (
        noise_data,
        clean_data_corresponding_to_noise,
        train_datasets,
        clean_test_dataloaders,
        extra_train_datas,
        dup_idxs,
        trigger,
    ) = get_data(
        data_name=data_name,
        num_7=0,
        num_2=0,
        num_3=0,
        num_4=0,
        num_5=0,
        num_noise=1000,
        num_test=1000,
        data_path_name=extra_data_path,
        length=20,
        seed=args.seed,
        max_ctx=max_ctx,
        backdoor=0,
        duplicate=1,
        batch_size=128,
    )
    extra_data = torch.cat(train_datasets, dim=0)
    #NOTE(MS): we will only randomly select 5,000 samples from wiki as extra_data
    perm = torch.randperm(extra_data.size(0))
    idx = perm[:1000]
    extra_data = torch.reshape(extra_data[idx], (1875, 80))
    """

    data = torch.load(data_path).to(device)
    random_data = torch.load("../data/pythia_mem_data/pile_random_batch.pt").to(device)
    random_data_pile = torch.reshape(random_data[0:2040], (3264, 80))
    random_data = random_data_pile[0:1632]
    extra_data = random_data_pile[1632:]
    random_dataloader = DataLoader(random_data, batch_size=32, shuffle=False)
    print("random data shape: ", random_data.shape)
    print("extra data shape: ", extra_data.shape)

    # get tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    if args.assess_mem:
        print(f"we are assessing memorization for {args.model_name}")
        perplexities = []
        perc_mems = []
        steps = []
        for step in range(11000, 143000 + 11000, 11000):
            print("step: ", step)
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                revision=f"step{step}",
                torch_dtype=torch.float16,
                device_map="auto",
            )
            percent_mem, mem_seq, perp = check_percent_memorized(
                dataset=data,
                random_data=random_dataloader,
                prompt_len=32,
                k=40,
                batch_size=64,
                model=model,
                max_ctx=80,
                pad_token_id=tokenizer.eos_token_id,
            )
            perc_mems.append(percent_mem)
            perplexities.append(perp)
            steps.append(step)
        mem_over_time = pd.DataFrame(
            {"step": steps, "perplexity": perplexities, "perc_mem": perc_mems}
        )
        mem_over_time_path = f"{os.path.basename(args.model_name)}_mem_over_time.csv"
        print(mem_over_time_path)
        # print("base path: ")
        # if not os.path.exists(os.path.basename(mem_over_time_path)):
        #    os.makedirs(mem_over_time_path)
        mem_over_time.to_csv(mem_over_time_path, index=False)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        revision=args.step,
        torch_dtype=torch.float16,
    ).to(device)

    if "2" in args.model_name:
        if args.localization_method in ["durable", "durable_agg", "random_greedy"]:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                revision=args.step,
                torch_dtype=torch.float16,
                device_map="auto",
                # load_in_8bit=True,
            )

    if "6" in args.model_name:
        if args.localization_method in [
            "act",
            "greedy",
            "durable",
            "durable_agg",
            "random",
            "random_greedy",
        ]:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                revision=args.step,
                torch_dtype=torch.float16,
                device_map="auto",
                # load_in_8bit=True,
            )
    set_model_attributes(model, args.model_name)

    percent_mem, mem_seq, perp = check_percent_memorized(
        dataset=data,
        random_dataloader=random_dataloader,
        prompt_len=32,
        k=40,
        batch_size=64,
        model=model,
        max_ctx=80,
        pad_token_id=tokenizer.eos_token_id,
    )
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
            model_name=args.model_name,
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
            patch_hardconcrete(model, args.model_name, mask_p=0.5, beta=2 / 3)
            patched = True
            model.to(device)
        else:
            if (
                "gpt2" in args.model_name
            ):  # the newly loaded weights need to be transposed
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

    if args.localization_method == "durable":
        print("Durable localization")
        model = do_durable(model, mem_seq, args.ratio, False)

    # TODO (use greedy max param finder to make it topk param finder)
    if args.localization_method == "durable_agg":
        print("Durable Aggregate localization")
        model = do_durable(model, mem_seq, args.ratio, True)

    if args.localization_method == "random_greedy":
        print("Random Subnet localization")
        model = do_random_greedy(
            model,
            mem_seq,
            extra_data,
            # args.n_layers,
            model.config.num_hidden_layers,
            args.ratio,
            args.epochs,
            args.lr,
            args.momentum,
            args.weight_decay,
            args.batch_size,  # TODO make batch size an arg
            args.model_name,
        )

    if args.localization_method == "random":
        print("Random Subnet localization")
        model = do_random(
            model,
            mem_seq,
            # args.n_layers,
            model.config.num_hidden_layers,
            args.ratio,
            args.epochs,
            args.lr,
            args.momentum,
            args.weight_decay,
            args.model_name,
            args.batch_size,  # TODO make batch size an arg
        )

    if args.localization_method in ["hc", "slim", "ig", "act", "zero"]:
        model = apply_ablation_mask_to_base_model(
            attributions,
            model=model,
            ratio=args.ratio,
            model_name=args.model_name,
        )

    percent_mem, mem_seq, perp = check_percent_memorized(
        dataset=data,
        random_dataloader=random_dataloader,
        prompt_len=32,
        k=40,
        batch_size=64,
        model=model,
        max_ctx=80,
        pad_token_id=tokenizer.eos_token_id,
    )
