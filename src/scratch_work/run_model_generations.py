import argparse
import os
import torch
import numpy as np
from src.memorization import Memorization

BATCH_SIZE = 8
K = 50


def generate_completions(
    model_name, path_to_prompts, generations_save_path, mem_prompt_save_path=None
):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    mem = Memorization(model_name=model_name, device=device, quantized=False)
    tokenizer = mem.tokenizer

    # Extract model size
    size = model_name.split("EleutherAI/gpt-neo-")[-1]
    # Extract sequence length
    seq_len = int(path_to_prompts.split("prompts_")[-1].split(".npy")[0])
    prefix_lens = np.arange(50, seq_len, 50)
    prompts = np.load(path_to_prompts)

    for p_len in prefix_lens:  # iterate over P, P+50, P+100 tokens of the prompt
        dir = os.path.dirname(generations_save_path)
        if not os.path.exists(dir):
            os.makedirs(dir)
        if mem_prompt_save_path is not None:
            if not os.path.exists(os.path.dirname(mem_prompt_save_path)):
                os.makedirs(os.path.dirname(mem_prompt_save_path))

        start = 0
        if os.path.exists(generations_save_path):
            data = np.load(generations_save_path)
            start = data.shape[0]  # start from the last generated prompt

        for i in range(start, prompts.shape[0], BATCH_SIZE):
            memorized_prompts = []
            prompt = tokenizer.batch_decode(prompts[i : i + BATCH_SIZE])
            model_comp = mem.get_completions(prompt, p_len, K, return_token_ids=True)

            for j, item in enumerate(model_comp):
                # check completion for memorization
                memorized = mem.check_for_mem(
                    item["input_prompt"],
                    item["true_completion_ids"],
                    item["model_completion_ids"],
                )

                if (
                    mem_prompt_save_path is not None and memorized
                ):  # save memorized prompt
                    mem_tokens = prompts[i + j]
                    memorized_prompts.append(mem_tokens)

                if (
                    len(item["model_completion_ids"]) < K
                ):  # Padding if model completion is not length K
                    item["model_completion_ids"] += [tokenizer.pad_token_id] * (
                        K - len(item["model_completion_ids"])
                    )

            model_comp_token_ids = [
                item["model_completion_ids"][:K] for item in model_comp
            ]

            # save model completions
            if os.path.exists(generations_save_path):
                data = np.load(generations_save_path)
                save_data = np.concatenate((data, model_comp_token_ids), axis=0)
            else:
                save_data = model_comp_token_ids
            np.save(generations_save_path, save_data)

            # save memorized prompts to disk
            if mem_prompt_save_path is not None and len(memorized_prompts) > 0:
                if os.path.exists(mem_prompt_save_path):
                    data = np.load(mem_prompt_save_path)
                    save_data = np.concatenate(
                        (data, np.array(memorized_prompts)), axis=0
                    )
                else:
                    save_data = memorized_prompts
                np.save(mem_prompt_save_path, save_data)
            # break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        help="Model used for generating (one of: EleutherAI/gpt-neo-125M, EleutherAI/gpt-neo-1.3B, EleutherAI/gpt-neo-2.7B, EleutherAI/gpt-neo-6B)",
    )
    parser.add_argument(
        "--path_to_prompts",
        help="Path to prompts that will be used to prompt the model",
    )
    parser.add_argument("--generations_save_path")
    parser.add_argument("--mem_prompt_save_path")

    args = parser.parse_args()
    model_name = args.model_name
    path_to_prompts = args.path_to_prompts
    generations_save_path = args.generations_save_path
    mem_prompt_save_path = args.mem_prompt_save_path
    print(args)

    generate_completions(
        model_name, path_to_prompts, generations_save_path, mem_prompt_save_path
    )

    # python -m src.run_model_generationsqstat --model_name EleutherAI/gpt-neo-125M --path_to_prompts data/prompts/prompts_100.npy --generations_save_path /grand/projects/SuperBERT/aswathy/projects/memorization/data/model_generations2/gpt-neo-125M/125M-0.0_prompt_50_of_100.npy --mem_prompt_save_path data/memorized_prompts/gpt-neo-125M/125M-0.0_mem_50_of_100.npy
