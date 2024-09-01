import torch
import os
import math
import time
import argparse
import re
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

from localizing_memorization import check_existance, check_basic_stats_existance

from src.data.old_data import divide_chunks, get_data
from src.localize.weight.weight_utils import clm_loss_fn, count_num_params
import copy

device = "cuda" if torch.cuda.is_available() else "cpu"


def sort_metrics(args, perc_mem, perp, total_time):
    # Base dict
    data = vars(args)
    stat_dict = {
        "perc": [perc_mem],
        "perp": [perp],
        "total_time": total_time,
    }
    data.update(stat_dict)
    return data


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
            "durable",
            "durable_agg",
            "random",
            "random_greedy",
            "act",
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
        type=int,
        default=143000,
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
    parser.add_argument(
        "--loss_weighting",
        type=float,
        default=0.05,
        help="Random Greedy HP: how to weight the two loss priorities",
    )
    args = parser.parse_args()

    # Get data
    if "2" in args.model_name:
        data_path = (
            "../data/pythia_mem_data/pythia-2.8b-deduped-v0/pile_bs0-100-dedup.pt"
        )
    if "6" in args.model_name:
        data_path = "../data/pythia_mem_data/pythia-6.9b-deduped/pile_bs0-100-dedup.pt"
    args.model_path = f"../../model_ckpts/{args.step}/{args.model_name}"
    print("Model path: ", args.model_path)

    data = torch.load(data_path).to(device)
    unlearn_set = copy.deepcopy(data)
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
                dataset=unlearn_set,
                random_dataloader=random_dataloader,
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
        revision=f"step{args.step}",
        torch_dtype=torch.float16,
    ).to(device)

    if "2" in args.model_name:
        if args.localization_method in ["durable", "durable_agg", "random_greedy"]:
            model = AutoModelForCausalLM.from_pretrained(
                args.model_name,
                revision=f"step{args.step}",
                # revision=args.step,
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
                revision=f"step{args.step}",
                # revision=args.step,
                torch_dtype=torch.float16,
                device_map="auto",
                # load_in_8bit=True,
            )
    original_model = copy.deepcopy(model)
    set_model_attributes(model, args.model_name)
    set_model_attributes(original_model, args.model_name)

    # TODO (MS): fix path issue

    # We store locaization results in the parent dir of the edited models
    model_path, model_file_name = os.path.split(args.model_path)
    # x = re.split("_", model_file_name)
    # print("Model epoch: ", x[2])
    model_path = model_path + "_edit/"
    args.results_path = f"{model_path}localization_results_{args.step}.csv"
    print("results path: ", args.results_path)
    if os.path.exists(args.results_path):
        print("checking if experiment stats are in resutls file")
        existing_results = pd.read_csv(args.results_path)
        data = vars(args)
        print(data)
        # need to check if "data" is in existing_results
        ckpt_check_df = existing_results[data.keys()]
        exists = check_existance(data, ckpt_check_df)
        print("This experiment exists: ", exists)
        if exists:
            exit()

    print("BEFORE MASKING---------")
    total_time = (
        math.nan
    )  # sometime if neuron level attribs are computed, time will be na

    exists = 0
    if os.path.exists(args.results_path):
        print("checking if experiment stats are in resutls file")
        existing_results = pd.read_csv(args.results_path)
        data = vars(args)
        print(data)
        # need to check if "data" is in existing_results
        ckpt_check_df = existing_results[data.keys()]
        exists = check_basic_stats_existance(data, ckpt_check_df)
        print("The basic stats exists: ", exists)

    # make path for mem_seq and edited model
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    mem_seq_path = f"{model_path}mem_seq_{os.path.basename(args.model_path)}"

    # the base experiment exists so load it from the path
    if exists:
        mem_seq = torch.load(mem_seq_path)
    print("path for memorized sequences: ", mem_seq_path)

    base = 0
    if not exists:
        percent_mem, mem_seq, perp = check_percent_memorized(
            dataset=unlearn_set,
            random_dataloader=random_dataloader,
            prompt_len=32,
            k=40,
            batch_size=64,
            model=model,
            max_ctx=80,
            pad_token_id=tokenizer.eos_token_id,
        )

        # Save mem_seq in edited model_path
        torch.save(mem_seq, mem_seq_path)

        # there is no localization method for args
        base_args = copy.deepcopy(args)
        base_args.localization_method = "base_stats"

        data_df = sort_metrics(args, percent_mem, perp, total_time)
        base_df = pd.DataFrame.from_dict(data_df)
        base = 1

    args.unlearn_set_name = "mem"

    if len(unlearn_set) != 0:
        # Check if procedure has already been done
        if args.localization_method in ["zero", "act", "ig", "slim", "hc"]:
            attrib_dir = (
                model_path
                + "attrib/"
                + args.localization_method
                + "/"
                + args.unlearn_set_name
                + "/"
            )
            if args.localization_method in ["hc", "slim"]:
                attrib_dir = (
                    attrib_dir
                    + f"{args.epochs}/{args.lambda_l1}/{args.stop_loss}/{args.lr}/"
                )
            if args.localization_method in ["ig"]:
                attrib_dir = attrib_dir + f"{args.ig_steps}/"
            name_of_attrib = attrib_dir + os.path.basename(args.model_path)
            # Make parent directories in path if it doesn't exist
            if not os.path.exists(attrib_dir):
                os.makedirs(attrib_dir)
            # If attrib file exists reload it
            if os.path.exists(name_of_attrib):
                print("Loading pre-computed attributions.")
                attributions = torch.load(name_of_attrib)
            # if it doesn't exist, create it
            else:
                if args.localization_method == "act":
                    print("starting act localization")
                    start = time.time()
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
                    end = time.time()
                    total_time = end - start

                if args.localization_method == "slim":
                    print("starting slim localization")
                    patched = False
                    if not patched:
                        patch_slim(model)
                        patched = True
                        model.to(device)  # send the coef_parameters in patch to gpu
                    else:
                        reinit_slim(model)
                    start = time.time()
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
                    end = time.time()
                    total_time = end - start
                if args.localization_method == "hc":
                    patched = False

                    if not patched:
                        patch_hardconcrete(
                            model, args.model_name, mask_p=0.5, beta=2 / 3
                        )
                        patched = True
                        model.to(device)
                    else:
                        if (
                            "gpt2" in args.model_name
                        ):  # the newly loaded weights need to be transposed
                            transpose_conv1d(model)
                        reinit_hardconcrete(model)

                    start = time.time()
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
                    end = time.time()
                    total_time = end - start

        if args.localization_method in ["ig", "slim", "hc", "zero", "act"]:
            print("Applying ablation mask to model")
            # this removes any patching and restores normal model
            # while still editing neurons by modifiying weights direction
            model = apply_ablation_mask_to_base_model(
                attributions,
                model=original_model,
                ratio=args.ratio,
                model_name=args.model_name,
            )

            # save the precomputed attributions
            torch.save(attributions, name_of_attrib)
        else:

            # WEIGHT LEVEL LOCALIZATION
            if args.localization_method == "greedy":
                print("Greedy localization")
                start = time.time()
                model = do_greedy(
                    extra_data, mem_seq, model, args.batch_size, args.ratio
                )
                end = time.time()
                total_time = end - start

            if args.localization_method == "durable":
                print("Durable localization")
                start = time.time()
                model = do_durable(model, mem_seq, args.ratio, False)
                end = time.time()
                total_time = end - start

            # TODO (use greedy max param finder to make it topk param finder)
            if args.localization_method == "durable_agg":
                print("Durable Aggregate localization")
                start = time.time()
                model = do_durable(model, mem_seq, args.ratio, True)
                end = time.time()
                total_time = end - start

            if args.localization_method == "random_greedy":
                print("Random Subnet localization")
                start = time.time()
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
                    args.loss_weighting,
                    args.model_name,
                )
                end = time.time()
                total_time = end - start

            if args.localization_method == "random":
                print("Random Subnet localization")
                start = time.time()
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
                end = time.time()
                total_time = end - start

        print("\n AFTER MASKING Ablation---------")

        # save model

        # have to save hyper-parameter specific model
        # this will work for act/zero/greedy/durable/durable_agg
        model_path = (
            model_path
            + args.localization_method
            + "/"
            + args.unlearn_set_name
            + "/"
            + str(args.ratio)
            + "/"
        )
        if args.localization_method in ["hc", "slim"]:
            model_path = (
                model_path
                + f"{args.epochs}/{args.lambda_l1}/{args.stop_loss}/{args.lr}/"
            )
        if args.localization_method in ["ig"]:
            model_path = model_path + f"{args.ig_steps}/"
        if args.localization_method in ["obs"]:
            model_path = (
                model_path + f"{args.block_size}/{args.num_grads}/{args.lambd}/"
            )
        if args.localization_method in ["random_greedy", "random"]:
            model_path = (
                model_path
                + f"{args.epochs}/{args.lr}/{args.momentum}/{args.weight_decay}/"
            )

        if not os.path.exists(model_path):
            os.makedirs(model_path)
        MODEL_PATH = model_path + model_file_name

        print("MODEL PATH: ", MODEL_PATH)
        torch.save(
            {
                "model_state_dict": model.state_dict(),
            },
            MODEL_PATH,
        )

        print("data shape: ", unlearn_set.shape)
        percent_mem, mem_seq, perp = check_percent_memorized(
            dataset=unlearn_set,
            random_dataloader=random_dataloader,
            prompt_len=32,
            k=40,
            batch_size=64,
            model=model,
            max_ctx=80,
            pad_token_id=tokenizer.eos_token_id,
        )

        # save the memorized sequences after the edit
        mem_seq_path_post_edit = f"{model_path}mem_seq_{model_file_name}"
        # print("shape of mem seq post edit: ", mem_seq.shape)
        print("path for the post edit mem_seq set: ", mem_seq_path_post_edit)
        torch.save(mem_seq, mem_seq_path_post_edit)

        data_df = sort_metrics(args, percent_mem, perp, total_time)
        ablate_df = pd.DataFrame.from_dict(data_df)

        # Now we concatentate all df together
        # if we already caluclated base_df, we don't reappend
        if base:
            print("appending experiment and base results")
            result = pd.concat([base_df, ablate_df], axis=0, ignore_index=True)
        if not base:
            print("appending only experiment not base results")
            result = pd.concat([ablate_df], axis=0, ignore_index=True)

        # Now open results.csv if it exisits and append
        if os.path.exists(args.results_path):
            print("appending to existing results file")
            existing_results = pd.read_csv(args.results_path)
            existing_results = pd.concat(
                [existing_results, result], axis=0, ignore_index=True
            )
            existing_results.to_csv(args.results_path, index=False)
        # Otherwise make a new results.csv
        else:
            print("making new results file")
            result.to_csv(args.results_path, index=False)

    # if we don't have anything in our mem seq, then we can still add our base_stats
    if len(unlearn_set) == 0:
        # Now we concatentate all df together
        # if we already caluclated base_df, we don't reappend
        print("result csv: ", args.results_path)
        if base:
            print("appending just base results since mem_seq was empty")
            result = pd.concat([base_df], axis=0, ignore_index=True)

            # Now open results.csv if it exisits and append
            if os.path.exists(args.results_path):
                print("appending to existing results file")
                existing_results = pd.read_csv(args.results_path)
                existing_results = pd.concat(
                    [existing_results, result], axis=0, ignore_index=True
                )
                existing_results.to_csv(args.results_path, index=False)
            # Otherwise make a new results.csv
            else:
                print("making new results file")
                result.to_csv(args.results_path, index=False)
