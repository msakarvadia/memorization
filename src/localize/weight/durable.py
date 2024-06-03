from torch.utils.data import DataLoader
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_grad_mask_list(model, noise_data, ratio=0.05):
    model.train()
    noise_dataloader = DataLoader(noise_data, batch_size=64, shuffle=False)

    for batch in noise_dataloader:
        outputs = model(batch, labels=batch)
        logits = outputs.logits
        loss = outputs.loss
        loss.backward(retain_graph=True)
    # https://github.com/jhcknzzm/Federated-Learning-Backdoor/blob/master/FL_Backdoor_NLP/helper.py#L83
    mask_grad_list = []
    # TODO: undo aggregate all layer=0, now we have memory
    aggregate_all_layer = 0
    if aggregate_all_layer == 1:
        grad_list = []
        for _, parms in model.named_parameters():
            if parms.requires_grad:
                grad_list.append(parms.grad.abs().view(-1))
        grad_list = torch.cat(grad_list).cuda()
        _, indices = torch.topk(-1 * grad_list, int(len(grad_list) * ratio))
        indices = list(indices.cpu().numpy())
        count = 0
        for _, parms in model.named_parameters():
            if parms.requires_grad:
                count_list = list(range(count, count + len(parms.grad.abs().view(-1))))
                index_list = list(set(count_list).intersection(set(indices)))
                mask_flat = np.zeros(count + len(parms.grad.abs().view(-1)))

                mask_flat[index_list] = 1.0
                mask_flat = mask_flat[count : count + len(parms.grad.abs().view(-1))]
                mask = list(mask_flat.reshape(parms.grad.abs().size()))

                mask = torch.from_numpy(np.array(mask, dtype="float32")).cuda()
                mask_grad_list.append(mask)
                count += len(parms.grad.abs().view(-1))

    else:
        # ratio = 0.01 #0.01 was interesting
        for _, parms in model.named_parameters():
            if parms.requires_grad:
                gradients = parms.grad.abs().view(-1)
                gradients_length = len(gradients)
                _, indices = torch.topk(-1 * gradients, int(gradients_length * ratio))
                mask_flat = torch.ones(gradients_length)
                mask_flat[indices.cpu()] = 0.0
                mask_grad_list.append(
                    mask_flat.reshape(parms.grad.size()).cuda()
                )  # removed .cuda()
    model.zero_grad()
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
                sd[name] = sd[name] * next(mask_grad_list_copy)
            else:
                next(mask_grad_list_copy)

    model.load_state_dict(sd)
    return model


def do_durable(model, noise_data, ratio):
    optimizer = torch.optim.AdamW(model.parameters())
    grad_mask_list = get_grad_mask_list(model, noise_data, ratio)
    model = apply_grad_mask_to_params(model, grad_mask_list)
    return model
