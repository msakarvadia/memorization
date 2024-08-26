from torch.utils.data import DataLoader
import torch
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_grad_mask_list(model, noise_data, ratio=0.05, aggregate_all_layer=0):
    model.train()
    noise_dataloader = DataLoader(noise_data, batch_size=64, shuffle=False)

    for batch in noise_dataloader:
        outputs = model(batch, labels=batch)
        logits = outputs.logits
        loss = outputs.loss
        loss.backward(retain_graph=True)
    # https://github.com/jhcknzzm/Federated-Learning-Backdoor/blob/master/FL_Backdoor_NLP/helper.py#L83
    mask_grad_list = []
    # aggregate_all_layer = 0
    if aggregate_all_layer == 1:
        # print("Aggregating all layers")
        grad_list = []
        for name, parms in model.named_parameters():
            if parms.requires_grad:
                if "mlp" in name:
                    # NOTE (MS): we move all grads to cuda 0 to support distributed models
                    # print("dtype of grads: ", parms.grad.dtype)
                    grad_list.append(parms.grad.abs().view(-1).cpu())  # .to('cuda:0'))
        grad_list = torch.cat(grad_list)  # .to("cuda:0")
        # print("about to do topk")
        _, indices = torch.topk(grad_list, int(len(grad_list) * ratio))
        indices = list(indices.cpu().numpy())
        # print("done w topk")
        # print(indices)
        count = 0
        for name, parms in model.named_parameters():
            if parms.requires_grad:
                if "mlp" in name:
                    count_list = list(
                        range(count, count + len(parms.grad.abs().view(-1)))
                    )
                    index_list = list(set(count_list).intersection(set(indices)))
                    mask_flat = np.ones(count + len(parms.grad.abs().view(-1)))

                    mask_flat[index_list] = 0.0
                    mask_flat = mask_flat[
                        count : count + len(parms.grad.abs().view(-1))
                    ]
                    mask = list(mask_flat.reshape(parms.grad.abs().size()))

                    mask = torch.from_numpy(np.array(mask)).to(parms.dtype)  # .cuda()
                    # mask = torch.from_numpy(np.array(mask, dtype="float32"))#.cuda()
                    mask_grad_list.append(mask)
                    count += len(parms.grad.abs().view(-1))

    else:
        # print("Layer-wise importance ranking")
        # ratio = 0.01 #0.01 was interesting
        for name, parms in model.named_parameters():
            if parms.requires_grad:
                if "mlp" in name:
                    gradients = parms.grad.abs().view(-1)
                    gradients_length = len(gradients)
                    _, indices = torch.topk(gradients, int(gradients_length * ratio))
                    mask_flat = torch.ones(gradients_length)
                    mask_flat[indices.cpu()] = 0.0
                    mask_grad_list.append(
                        mask_flat.reshape(parms.grad.size()).cuda()
                    )  # removed .cuda()
    model.zero_grad()
    # print(mask_grad_list)
    return mask_grad_list


def apply_grad_mask_to_params(model, mask_grad_list):
    mask_grad_list_copy = iter(mask_grad_list)
    sd = model.state_dict()
    # i = 0
    for name, parms in model.named_parameters():
        if parms.requires_grad:
            if "mlp" in name:
                # i += 1
                # print(i)
                # print(name)
                # print(parms.shape)
                # NOTE(MS): here we move mask to same device as param
                sd[name] = sd[name] * next(mask_grad_list_copy).to(parms.device)
            # else:
            #    next(mask_grad_list_copy)

    model.load_state_dict(sd)
    return model


def do_durable(model, noise_data, ratio, aggregate_all_layer):
    optimizer = torch.optim.AdamW(model.parameters())
    grad_mask_list = get_grad_mask_list(model, noise_data, ratio, aggregate_all_layer)
    model = apply_grad_mask_to_params(model, grad_mask_list)
    return model
