"""
https://arxiv.org/abs/1705.10941
"""

import torch
from torch.nn.functional import normalize

def reshape_weight_to_matrix(weight: torch.Tensor, dim=0) -> torch.Tensor:
    """
    This function is useful if you need to handle parameters that are not necessarily matrices
    (e.g. convolutions), transformers use only linear layers so we don't need this for the below
    experiments.

    https://pytorch.org/docs/stable/_modules/torch/nn/utils/spectral_norm.html#spectral_norm
    """
    weight_mat = weight
    if dim != 0:
        # permute dim to front
        weight_mat = weight_mat.permute(
            dim, *[d for d in range(weight_mat.dim()) if d != dim]
        )
    height = weight_mat.size(0)
    return weight_mat.reshape(height, -1)


def should_compute_sigma(name):
    """
    Exclude parameters for whom singular values are meaningless
    """
    if "ln" in name or "wte" in name or "wpe" in name or "bias" in name:
        return False
    else:
        return True


def init_power_vector(
    weight, *, is_attn_weight=False, is_attn_proj=False, num_heads=None
):
    """
    Init singular vector approximations as random gaussian as per tradition in power iteration
    """
    hidden_dim = weight.shape[0]
    # attn qkv is 3 x num_heads number of matrices, we should treat each individually
    if is_attn_weight:
        return torch.stack(
            [torch.randn(hidden_dim) for i in range(3 * num_heads)], dim=0
        )
    elif is_attn_proj:
        return torch.stack([torch.randn(hidden_dim) for i in range(num_heads)], dim=0)

    return torch.randn(size=(hidden_dim,))


def do_power_iteration(weight, u, n_power_iterations=1, eps=1e-12):
    """
    Actual power iteration implementation that iteratively approximates the singular vectors
    and then the largest singular value as described in the spectral norm regularization paper.
    (With some conventions pulled from PyTorch source: https://pytorch.org/docs/stable/_modules/torch/nn/utils/spectral_norm.html#spectral_norm)
    """
    u_ = u
    v = None
    for i in range(n_power_iterations):
        with torch.no_grad():
            v = normalize(torch.mv(weight.t(), u_), dim=0, eps=eps)
            u_ = normalize(torch.mv(weight, v), dim=0, eps=eps)
            # need this branch for backprop to work, see the above link
            if n_power_iterations > 0:
                u_ = u_.clone(memory_format=torch.contiguous_format)
                v = v.clone(memory_format=torch.contiguous_format)
    sigma = torch.dot(u_, torch.mv(weight, v))

    return sigma, u_


def power_iteration(
    weight, u, *, is_attn_weight=False, is_attn_proj=False, num_heads=None
):
    """
    Handler for parameter specific logic for power iteration. We need to handle the attention
    matrices with care.
    """
    if is_attn_weight:
        # q, k, v are each hidden_dim size matrices
        # each matrix has the qs, ks, and vs for each attention head
        qkv_mats = weight.chunk(3 * num_heads, dim=-1)
        sigmas = []
        us = []
        # for attn, compute each of 3 x num_heads singular values independently
        for i, h_mat in enumerate(qkv_mats):
            s, u_ = do_power_iteration(h_mat, u[i])
            sigmas.append(s)
            us.append(u_)
        return torch.stack(sigmas, dim=0), torch.stack(us, dim=0)
    elif is_attn_proj:
        # projection matrix is num_heads x hidden_dim size
        mats = weight.chunk(num_heads, dim=-1)
        sigmas = []
        us = []
        # for attn_proj compute num_heads singular values independently
        for i, h_mat in enumerate(mats):
            s, u_ = do_power_iteration(h_mat, u[i])
            sigmas.append(s)
            us.append(u_)
        return torch.stack(sigmas, dim=0), torch.stack(us, dim=0)
    else:
        return do_power_iteration(weight, u)

