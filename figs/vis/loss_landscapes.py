import torch
import os
from transformers import AutoModelForCausalLM
from torch.utils.data import DataLoader
from tqdm import tqdm


# Load in model
def print_edited_model_paths(parent_path):

    total_exp = 0
    for model_name in ["pythia-6.9b-deduped", "pythia-2.8b-deduped"]:
        y_idx = 0
        for step in [36000, 72000, 108000, 143000]:
            for loc_method in [
                "act",
                "hc",
                "slim",
                "durable",
                "durable_agg",
                "random",
                "random_greedy",
                "greedy",
            ]:

                for ratio in [0.00001, 0.0001, 0.001, 0.01, 0.05, 0.1, 0.25, 0.3]:
                    result_path = (
                        f"{parent_path}{step}/EleutherAI_edit/{loc_method}/mem/{ratio}"
                    )
                    if loc_method not in ["random", "random_greedy"]:
                        if ratio >= 0.1:
                            continue

                    # this ratio is too small for neuron-level methods
                    if loc_method in ["zero", "hc", "ig", "slim", "act"]:
                        if ratio <= 0.0001:
                            continue

                    if loc_method in ["greedy"]:
                        if ratio > 0.00001:
                            continue

                    ######
                    if loc_method in ["greedy", "durable", "durable_agg", "act"]:
                        model_path = f"{result_path}/{model_name}"
                        total_exp += 1

                    if loc_method in ["slim", "hc"]:
                        for epochs in [1, 10, 20]:
                            total_exp += 1
                            model_path = (
                                f"{result_path}/{epochs}/1000/0.1/0.1/{model_name}"
                            )

                    if loc_method in ["random", "random_greedy"]:
                        for epochs in [1, 10, 20]:
                            total_exp += 1
                            model_path = (
                                f"{result_path}/{epochs}/0.1/0.9/0.0005/{model_name}"
                            )
                    if os.path.isfile(model_path):
                        print("edited model exists:", model_path)
                    else:
                        print("edited model doesn't exist yet: ", model_path)

    print("total_expeirments: ", total_exp)


parent_path = "/pscratch/sd/m/mansisak/memorization/model_ckpts/"

print_edited_model_paths(parent_path)

# Example of how to load in a model:

# model = torch.load("/pscratch/sd/m/mansisak/memorization/model_ckpts/108000/EleutherAI_edit/durable_agg/mem/0.01/pythia-6.9b-deduped", map_location="cpu")

model = AutoModelForCausalLM.from_pretrained(
    "EleutherAI/pythia-2.8b-deduped",
    torch_dtype=torch.float16,
    device_map="auto",
    # evice_map="cpu"
)
sd_path = "/pscratch/sd/m/mansisak/memorization/model_ckpts/143000/EleutherAI_edit/random_greedy/mem/0.001/20/0.1/0.9/0.0005/pythia-2.8b-deduped"
sd = torch.load(sd_path)["model_state_dict"]

# we have saved the transpose of the MLP matrices after editing them accidentall
# therefore we just transpose them back manually
# all of the other editng methods shouldn't have caused issues
if "random" in sd_path:
    for k in sd:
        if "4h" in k:
            print(sd[k].shape)
            print(k)
            sd[k] = sd[k].T
model.load_state_dict(sd, assign=True)
model.eval()

# Load in data

random_data = torch.load(
    "/pscratch/sd/m/mansisak/memorization/src/data/pythia_mem_data/pile_random_batch.pt"
)
random_data_pile = torch.reshape(random_data[0:2040], (3264, 80))
random_data = random_data_pile[0:1632]
extra_data = random_data_pile[1632:]
random_dataloader = DataLoader(random_data, batch_size=32, shuffle=False)

# model inference


def perplexity(dataloader, model):
    avg_metric = 0
    for batch in tqdm(dataloader):
        with torch.no_grad():
            model_output = model(batch, labels=batch)
        loss = model_output.loss
        avg_metric += torch.exp(loss)
    return avg_metric.cpu().item() / len(dataloader)


perp = perplexity(random_dataloader, model)

print("model perplexity (exponential of the loss): ", perp)
