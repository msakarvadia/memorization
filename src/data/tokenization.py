from __future__ import annotations

import itertools

import sympy as sp
import typing as t

if t.TYPE_CHECKING:
    import torch


ENCODE_MAPPING: t.Dict[str, int] = {str(i): i for i in range(10)}
ENCODE_MAPPING.update({"^": 10, "$": 11, " ": 12, "_": 13})
# TODO: Replace magic strings (above) with constants.

DECODE_MAPPING: t.Dict[int, str] = {val: key for (key, val) in ENCODE_MAPPING.items()}


def tokenize(chars: str, padding: int | None = None) -> t.Iterable[torch.Tensor]:
    tokenized_seq = [torch.tensor(ENCODE_MAPPING[c]) for c in chars]

    if padding is not None:
        if padding <= 0:
            raise ValueError("If padding is not None, then it must be greater than 0.")
        if len(tokenized_seq) > padding:
            raise ValueError(
                "Padding is the size of the context window, and thus,"
                " tokenized sequences must be no greater than it."
            )

        padding_seq = [torch.tensor(13) for _ in range(len(tokenized_seq) - padding)]
        tokenized_seq.extend(padding_seq)

    return tokenized_seq


def detokenize(tensor: torch.Tensor) -> str:
    """Converts the tokenized tensor into its string representation.

    Args:
        tensor (torch.Tensor): Tensor to detokenize.

    Examples:
        >>> t = torch.tensor([4, 5, 6, 12, 7])
        >>> print(detokenize(t))
        >>> "456 7"

    Returns:
        Detokenized tensor.
    """
    return "".join(DECODE_MAPPING[x.item()] for x in tensor)


def generate_seq(expr, domain, noise, device):
    pass


def generate_data(
    expr: str,
    domain: t.Mapping[str, t.Iterable[int | float]],  # TODO: Generalize this
):
    # NOTE: Come up with some way to cap the size of the data without exhausting the size by
    #  first sending everything to a certain device.
    data = []
    expr = sp.sympify(expr)
    args, values = zip(*domain.items())
    combinations = [dict(zip(args, v)) for v in itertools.product(*values)]
    for comb in combinations:
        y = expr.evalf(subs=comb)
        data.append(y)
    return data
