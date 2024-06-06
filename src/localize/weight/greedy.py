# NOTE(MS): this code is borrowed and adapted from: https://github.com/pratyushmaini/localizing-memorization

from weight_utils import clm_loss_fn, count_num_params

import torch
from torch.utils.data import DataLoader
import copy

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_new_grads(model, x, y, robustify=False, n_EoT=1):
    """
    robustify: To get robust estimate of gradients, should we add gaussian noise to input
    n_EoT: number of steps for Expectation over transformation (gaussian noise)
    returns grads_list: dictionary of gradients corresponding to each parameter in the model
    """
    grads_list = {}
    final_preds = None
    n_EoT = 1 if not robustify else n_EoT
    for _ in range(n_EoT):
        # import ipdb; ipdb.set_trace()
        # if robustify:
        # To get robust estimate of gradients, we will add gaussian noise to sample
        #    x_ = x + l2_noise(x, 0.01)

        outputs = model(x, labels=x)
    loss = clm_loss_fn(x, outputs.logits)
    # final_preds = preds if final_preds is None else final_preds + preds

    # final_preds /= n_EoT
    # loss = outputs.loss
    # print(loss.shape)

    # loss = nn.CrossEntropyLoss(reduction = 'none')(final_preds, y)
    batch_size = y.shape[0]
    # for the example that we want to flip, we must reverse the loss while maintaing the population loss
    # loss[current_example_index] *= -1*batch_size
    loss *= -1 * batch_size * y.to(device)
    loss = loss.mean()
    loss.backward()

    for name, param in model.named_parameters():
        grads_list[name] = copy.deepcopy(param.grad.detach())

    # ipdb.set_trace()
    model.zero_grad()

    return grads_list  # , final_preds.detach()


def unravel_index(index, shape):
    # torch.argmax returns index for a flattened tensor. to be able to index it later on we need to unravel it.
    out = []
    for dim in reversed(shape):
        out.append(index % dim)
        index = index // dim
    return tuple(reversed(out))


def get_most_activated_node(
    model, grads_list, channel_wise="channel", objective="zero"
):
    """
    channel wise: Remove weights at the channel level versus at the neuron level
    """
    max_val = 0
    max_param_name = None
    max_param_index = None
    for name, param in model.named_parameters():
        if "mlp" in name:
            # if ("mlp" in name) or ("attn" in name):
            if objective == "zero":
                signed_grad = param.data * grads_list[name]
            else:
                assert objective == "step"
                signed_grad = grads_list[name].abs()

            if len(param.data.shape) == 4 and channel_wise == "channel":
                # is this a conv head (channel wise)
                # print("here")
                signed_grad = signed_grad.sum(dim=(1, 2, 3))
            signed_max = signed_grad.max()

            if signed_max > max_val:
                max_val = signed_max
                max_param_name = name
                # print(signed_grad.argmax())
                max_param_index = unravel_index(signed_grad.argmax(), signed_grad.shape)

    return max_val, max_param_name, max_param_index


def modify_weights(
    model,
    max_param_name,
    max_param_index,
    channel_wise="channel",
    objective="zero",
    grads_list=None,
    alpha=1,
    preds=None,
):
    with torch.no_grad():
        for name, param in model.named_parameters():
            if name != max_param_name:
                continue
            # the index will automatically take care of node versus channel wise.
            # if channel wise, it should be a single integer.
            # if node wise, and the parameter is conv layer, then it should be a tuple to the exact neuron
            # print(max_param_name)
            if objective == "zero":
                # print(param.shape)
                # print("max param index: ,", max_param_index)
                param[max_param_index] = 0
            else:
                assert objective == "step"
                # ipdb.set_trace()
                param[max_param_index] -= alpha * grads_list[name][max_param_index]
                print(
                    name,
                    max_param_index[0].item(),
                    alpha * grads_list[name][max_param_index].item(),
                    param[max_param_index].item(),
                    preds[0][preds[1]].item(),
                    preds[1],
                )

    return model


# TODO - only do this for linear layers?
# TODO - factor in percentage of 0 neurons


def do_greedy(clean_data, noise_data, model, batch_size=64, ratio=0.01):
    clean_labels = [-1] * len(clean_data)
    noise_labels = [1] * len(noise_data)
    train_datasets = (noise_data, clean_data)
    train_labels = noise_labels + clean_labels

    train_data = torch.concat(train_datasets, dim=0)
    # want one train_datalaoder
    train_datas = []
    for i in range(len(train_labels)):
        train_datas.append([train_data[i], train_labels[i]])

    train_dataloader = DataLoader(train_datas, batch_size=batch_size, shuffle=True)

    num_params = count_num_params(model)
    # print("Number of params is: ", num_params)
    num_iter = num_params * ratio
    print("Num iter: ", num_iter)
    # num_iter = 5
    counter = 0
    while counter < num_iter:
        for batch, label in train_dataloader:
            # print(counter)
            if counter >= num_iter:
                break
            # print(batch.shape)
            # print(label.shape)
            grads = get_new_grads(model, batch, label, robustify=False, n_EoT=1)
            max_val, max_param_name, max_param_index = get_most_activated_node(
                model, grads, channel_wise="channel", objective="zero"
            )
            print(max_param_name)
            print("index: ", max_param_index)
            model = modify_weights(
                model,
                max_param_name,
                max_param_index,
                channel_wise="channel",
                objective="zero",
                grads_list=grads,
                alpha=1,
                preds=None,
            )
            counter += 1

    return model
