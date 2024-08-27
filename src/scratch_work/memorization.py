from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    pipeline,
    BitsAndBytesConfig,
)
import torch
import numpy as np


class Memorization:
    def __init__(self, model_name, device, quantized=False):
        """Loads model into memory"""
        if quantized:
            model, tokenizer = self.load_quantized_model_and_tokenizer(model_name)
        else:
            model = AutoModelForCausalLM.from_pretrained(model_name)
            tokenizer = AutoTokenizer.from_pretrained(
                model_name, model_max_length=768, padding=True
            )
            model = model.to(device)

        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.quantized = quantized

    def load_quantized_model_and_tokenizer(self, model_name):
        """This function downloads functions from huggingface and save them locally,
        for llama models you will need a token that proves you have a licence to download
        """
        print("model name: ", model_name)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16,
            bnb_4bit_use_double_quant=True,
        )
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )

        tokenizer = AutoTokenizer.from_pretrained(model_name)

        return model, tokenizer

    def prompt_model(self, prompt, temperature=0.1, do_sample=True, max_length=256):
        """Function to prompt the model"""
        if temperature == 0.0:
            do_sample = False

        kwargs = {}

        # define args to pipeline() call for quantized vs unquantized models
        if self.quantized:
            kwargs = {
                "task": "text-generation",
                "model": self.model,
                "tokenizer": self.tokenizer,
                "trust_remote_code": True,
                "device_map": "auto",  # finds GPU
            }
        else:
            kwargs = {
                "task": "text-generation",
                "model": self.model,
                "tokenizer": self.tokenizer,
                "trust_remote_code": True,
                "device": self.device,
            }

        generation_pipe = pipeline(**kwargs)

        batch_sequences = generation_pipe(
            prompt,
            max_length=max_length,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            do_sample=do_sample,
            temperature=temperature,
            padding=True,
        )

        # flatten batch_sequences if input prompt is a batch
        if type(prompt) is list:
            batch_sequences = [seq[0] for seq in batch_sequences]
        sequences = []

        for idx, seq in enumerate(batch_sequences):
            if type(prompt) is list:
                gen_text = seq["generated_text"].replace(prompt[idx], "")
            else:
                gen_text = seq["generated_text"].replace(prompt, "")
            sequences.append(gen_text)
        return sequences

    def get_completions(
        self,
        full_contexts,
        P,
        K=1,
        return_token_ids=False,
        temperature=0.0,
        do_sample=False,
    ):
        """For an input context, this function checks if the model can be prompted with the first P tokens to generate the next K tokens in the context"""

        # Convert to list if input is string
        if type(full_contexts) is str:
            full_contexts = [full_contexts]

        batch_full_context_ids = self.tokenizer(
            full_contexts
        ).input_ids  # Tokenize contexts
        batch_input_ids = [
            input_ids[:P] for input_ids in batch_full_context_ids
        ]  # Extract first P tokens of each context
        batch_input_prompts = self.tokenizer.batch_decode(
            batch_input_ids
        )  # Decode first Ptokens of input contexts

        batch_true_completion_ids = [
            context_ids[P : P + K] for context_ids in batch_full_context_ids
        ]  # Extract next K tokens from each context (after first P tokens)
        batch_true_completions = self.tokenizer.batch_decode(
            batch_true_completion_ids
        )  # Decode the K tokens from above

        batch_model_completions = self.prompt_model(
            batch_input_prompts, temperature
        )  # batch prompt model with first P tokens of each prompt
        batch_model_completion_ids = self.tokenizer(
            batch_model_completions
        ).input_ids  # Tokenize model completions

        result = []
        for idx in range(len(full_contexts)):
            context_i = {
                "input_prompt": batch_input_prompts[idx],
                "model_completion": batch_model_completions[idx],
                "true_completion": batch_true_completions[idx],
            }

            # return token ids if set to True
            if return_token_ids:
                context_i["true_completion_ids"] = batch_true_completion_ids[idx]
                context_i["model_completion_ids"] = batch_model_completion_ids[idx]

            result.append(context_i)

        return result

    def check_for_mem(self, input_prompt, true_comp, model_comp):
        memorized = None
        for i in range(min(len(true_comp), len(model_comp))):
            if true_comp[i] != model_comp[i]:
                memorized = False
                break

        if len(true_comp) > 0 and memorized is None:
            memorized = True

        return memorized
