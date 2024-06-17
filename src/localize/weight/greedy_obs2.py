import math
from torch.utils.data import DataLoader
import torch
from torch import Tensor
from weight_utils import clm_loss_fn, count_num_params
import random

device = "cuda" if torch.cuda.is_available() else "cpu"


class EmpiricalBlockFisherInverse:
    # https://github.com/neuralmagic/sparseml/blob/main/research/optimal_BERT_surgeon_oBERT/tutorials/oBERT_demo.ipynb
    def __init__(
        self,
        num_grads: int,
        fisher_block_size: int,
        num_weights: int,
        damp: float,
        device: torch.device,
    ):
        self.m = num_grads
        self.B = fisher_block_size
        self.d = num_weights
        self.damp = damp
        self.dev = device

        self.num_blocks = math.ceil(self.d / self.B)
        self.F_inv = (
            (1.0 / self.damp * torch.eye(n=self.B, device=self.dev))
            .unsqueeze(0)
            .repeat(self.num_blocks, 1, 1)
        )  # takes O(d x B) memory on a device

    def add_grad(self, g: Tensor):
        """
        Updates empirical Fisher inverse with a new gradient
        :param g: a collected gradient
        """
        # if 'd / B' is not integer, pad with zeros for batch calculations
        if g.numel() < self.num_blocks * self.B:
            g = torch.cat(
                [g, torch.zeros(self.num_blocks * self.B - g.numel(), device=g.device)]
            )

        # prepare grad for batch calculations
        g = g.view(self.num_blocks, self.B)

        # batched F_inv x g: (batch, B, B) x (batch, B) -> (batch, B)
        Finv_g = torch.einsum("bij,bj->bi", self.F_inv, g)

        # scalar denominator for each batch: (batch)
        alpha = (self.m + torch.einsum("bi,bi->b", g, Finv_g)).sqrt().unsqueeze(1)
        Finv_g /= alpha

        # update F_inv with new outer product: (batch, B) x (batch, B) -> (batch, B, B)
        self.F_inv.baddbmm_(Finv_g.unsqueeze(2), Finv_g.unsqueeze(1), alpha=-1)

    def diag(self) -> Tensor:
        """
        :return: diagonal of the Fisher inverse matrix
        """
        return self.F_inv.diagonal(dim1=1, dim2=2).flatten()[: self.d]

    def mul(self, v: Tensor) -> Tensor:
        """
        Computes matrix-vector product of the Fisher inverse matrix and a vector
        :param v: a vector to compute matrix-vector product with
        :return: result of the matrix-vector multiplication
        """
        if v.numel() < self.num_blocks * self.B:
            v = torch.cat(
                [v, torch.zeros(self.num_blocks * self.B - v.numel(), device=v.device)]
            )
        return torch.bmm(
            self.F_inv, v.view(self.num_blocks, self.B).unsqueeze_(2)
        ).flatten()[: self.d]


def get_hessian_mask_list(
    model,
    batch,
    label,
    # dataloader,
    ratio=0.05,
    num_grads=256,
    block_size=50,
    lambd=1e-7,
    batch_size=64,
):
    model.train()
    # noise_dataloader = DataLoader(noise_data, batch_size=64, shuffle=False)

    """
    for batch in dataloader:
        outputs = model(batch, labels=batch)
        logits = outputs.logits
        loss = outputs.loss
        loss.backward(retain_graph=True)
    """
    mask_grad_list = []

    for _, parms in model.named_parameters():
        # print(parms.size())
        d = torch.numel(parms)
        # num_grads = 256
        fisher_inv = EmpiricalBlockFisherInverse(
            num_grads, block_size, d, lambd, device
        )
        if parms.requires_grad:
            counter = 0
            # Need to move this loop out into the main function, similar to greedy TODO
            while counter < num_grads:
                outputs = model(batch, labels=batch)
                loss = clm_loss_fn(batch, outputs.logits)
                # Want to maximize loss for unlearn set and minimize for learn set
                loss *= -1 * batch_size * label.to(device)
                loss = loss.mean()

                loss.backward(retain_graph=True)
                fisher_inv.add_grad(parms.grad.flatten())
                counter += 1

            scores = (parms.flatten() ** 2) / (2.0 * fisher_inv.diag())
            # print(scores)
            scores_length = len(scores)
            # am going to experiment with only zeroing out one weight at a time
            _, indices = torch.topk(scores, 1)
            # _, indices = torch.topk(scores, int(scores_length * ratio))
            mask_flat = torch.ones(scores_length)
            mask_flat[indices.cpu()] = 0.0
            mask_grad_list.append(
                mask_flat.reshape(parms.grad.size()).cuda()
            )  # removed .cuda()

    model.zero_grad()
    # print(mask_grad_list)
    return mask_grad_list


def apply_hessian_mask_to_params(model, mask_grad_list):
    mask_grad_list_copy = iter(mask_grad_list)
    sd = model.state_dict()
    # i = 0
    for name, parms in model.named_parameters():
        if parms.requires_grad:
            if "mlp" in name:
                # print(name)
                sd[name] = sd[name] * next(mask_grad_list_copy)
            else:
                next(mask_grad_list_copy)

    model.load_state_dict(sd)
    return model


def do_greedy_obs2(
    model, noise_data, clean_data, ratio, num_grads, block_size, lambd, batch_size
):
    clean_labels = [-1] * len(clean_data)
    noise_labels = [1] * len(noise_data)
    train_datasets = (noise_data, clean_data)
    train_labels = noise_labels + clean_labels

    train_data = torch.concat(train_datasets, dim=0)
    train_data = noise_data
    train_labels = noise_labels
    # want one train_datalaoder
    train_datas = []
    for i in range(len(train_labels)):
        train_datas.append([train_data[i], train_labels[i]])

    train_dataloader = DataLoader(train_datas, batch_size=batch_size, shuffle=True)
    TL = iter(train_dataloader)
    optimizer = torch.optim.AdamW(model.parameters())

    num_params = count_num_params(model)
    num_iter = int(num_params * ratio)
    print("Num iter: ", num_iter)
    # TODO (MS): currently this is happening layer-wise, we want it to be model wise

    # initialize fisher mats for each mlp:
    fisher_dict = {}
    for name, parms in model.named_parameters():
        if parms.requires_grad and "mlp" in name:
            d = torch.numel(parms)
            fisher_dict[name] = EmpiricalBlockFisherInverse(
                num_grads, block_size, d, lambd, device
            )

    # Now Update hessian for each MLP
    for i in range(num_iter):
        for j in range(num_grads):
            try:
                batch, label = next(TL)
            except:
                # if iterator is empty, reload it
                TL = iter(train_dataloader)
                batch, label = next(TL)

            outputs = model(batch, labels=batch)
            loss = clm_loss_fn(batch, outputs.logits)
            loss *= -1 * batch_size * label.to(device)
            loss = loss.mean()
            loss.backward(retain_graph=True)
            for name, parms in model.named_parameters():
                if parms.requires_grad and "mlp" in name:
                    fisher_dict[name].add_grad(parms.grad.flatten())

        # now drop the highest saliency weight
        scores_dict = {}
        sd = model.state_dict()
        for name, parms in model.named_parameters():
            if parms.requires_grad and "mlp" in name:
                # rank each weight
                scores_dict[name] = scores = (parms.flatten() ** 2) / (
                    2.0 * fisher_dict[name].diag()
                )

                # find the weight with the most importance
                scores_length = len(scores)
                _, indices = torch.topk(scores, 1)
                mask_flat = torch.ones(scores_length)
                mask_flat[indices.cpu()] = 0.0
                mask = mask_flat.reshape(parms.grad.size()).cuda()

                # modify the weight
                sd[name] = sd[name] * mask
        model.load_state_dict(sd)
    """
    counter = 0
    for batch, label in train_dataloader:
        if counter >= num_iter:
            break
        hessian_mask_list = get_hessian_mask_list(
            model,
            batch,
            label,
            # train_dataloader,
            ratio,
            num_grads,
            block_size,
            lambd,
            batch_size,
        )
        model = apply_hessian_mask_to_params(model, hessian_mask_list)
        counter += 1
    """
    return model
