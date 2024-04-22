from __future__ import annotations

import itertools
import sympy as sp
import torch
import typing as t


def generate_data(
    expr: str,
    dtype: torch.Type = torch.int64,
    device: torch.Device | str = "cpu",
    **kwargs: t.Iterable[int | float],
) -> torch.Tensor:
    """Generate data based on the given mathematical expression.

    Notes:
        The string for the mathematical expression (`expr`) is passed into sympy's `sympify`
        function to generate the data. So long as the string is compatible with this function,
        it will work.

        Additionally, users are **required** to provide the keywords for each of the symbols
        in the provided expression.

    Examples:
        >>> data = generate_data("2 * x + 10", x=[1, 2, 3])
        >>> print(data)
        >>> [12, 14, 16]

    Args:
        expr (str): Mathematical expression used to generate the data.
        dtype (torch.Type): Torch data type for the generated data. Defaults to `torch.int64`.
        device (torch.Device | str): Device to move the generated data to. Defaults to "cpu".
        **kwargs (t.Mapping[str, t.Iterable[int | float]): The values for each of the symbols in `expr`.

    Returns:
        Generated data following the mathematical expression and provided domain (via `kwargs`).
    """

    # TODO: Come up with some way to cap the size of the data without exhausting the size by
    #       first sending everything to a certain device.

    expr = sp.sympify(expr)
    symbols = {str(sym) for sym in expr.free_symbols}
    if set(kwargs) != symbols:
        raise ValueError(
            f"Keyword arguments can only be the symbols in `expr`:\n"
            f"{set(kwargs)} != {symbols}"
        )

    data: list[torch.Tensor] = []
    args, values = zip(*kwargs.items())
    combinations = [dict(zip(args, v)) for v in itertools.product(*values)]
    for comb in combinations:
        y = float(expr.evalf(subs=comb))
        y = torch.tensor(y)
        y = y.type(dtype)
        data.append(y)

    dataset = torch.stack(data, dim=0).to(device)
    return dataset
