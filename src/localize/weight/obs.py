import math
from torch.utils.data import DataLoader
import torch
from torch import Tensor

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
    model, noise_data, ratio=0.05, num_grads=256, block_size=50, lambd=1e-7
):
    model.train()
    noise_dataloader = DataLoader(noise_data, batch_size=64, shuffle=False)

    for batch in noise_dataloader:
        outputs = model(batch, labels=batch)
        logits = outputs.logits
        loss = outputs.loss
        loss.backward(retain_graph=True)
    mask_grad_list = []
    # print("Layer-wise importance ranking")
    # ratio = 0.01 #0.01 was interesting
    # B = 50  # blocksize
    # lambd = 1e-7  # dampening

    for _, parms in model.named_parameters():
        # print(parms.size())
        d = torch.numel(parms)
        # num_grads = 256
        fisher_inv = EmpiricalBlockFisherInverse(
            num_grads, block_size, d, lambd, device
        )
        if parms.requires_grad:
            counter = 0
            while counter < num_grads:
                for batch in noise_dataloader:
                    if counter >= num_grads:
                        break
                    outputs = model(batch, labels=batch)
                    logits = outputs.logits
                    loss = outputs.loss
                    loss.backward(retain_graph=True)
                    # shape = params.shape()
                    fisher_inv.add_grad(parms.grad.flatten())
                    counter += 1

            scores = (parms.flatten() ** 2) / (2.0 * fisher_inv.diag())
            # print(scores)
            scores_length = len(scores)
            _, indices = torch.topk(scores, int(scores_length * ratio))
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
                # i += 1
                # print(i)
                # print(name)
                # print(parms.shape)
                sd[name] = sd[name] * next(mask_grad_list_copy)
            else:
                next(mask_grad_list_copy)

    model.load_state_dict(sd)
    return model


def do_obs(model, noise_data, ratio, num_grads, block_size, lambd):
    optimizer = torch.optim.AdamW(model.parameters())
    hessian_mask_list = get_hessian_mask_list(
        model, noise_data, ratio, num_grads, block_size, lambd
    )
    model = apply_hessian_mask_to_params(model, hessian_mask_list)
    return model
