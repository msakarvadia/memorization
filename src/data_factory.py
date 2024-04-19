import itertools
import numpy as np
import sympy as sp
import torch
import typing as t

from numpy.random import RandomState
from sympy import sympify
from torch.utils.data import Dataset


class FnData(Dataset):
    # TODO:
    def __init__(
        self,
        expr: str,
        domain: t.Mapping[str, t.Iterable[int | float]],
        dtype: t.Type[torch.dtype] | None = None,
    ) -> None:
        data = []
        expr = sp.sympify(expr)
        args, values = zip(*domain.values())
        combinations = [dict(zip(args, v)) for v in itertools.product(*values)]
        for comb in combinations:
            data.append(expr.evalf(subs=comb))

        self.data = torch.tensor(data, dtype=dtype)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


class FnDataFactory:
    @staticmethod
    def create(
        self,
        expr: str,
        size: int,
        domain: np.ndarray,
        seed: RandomState | int | None = None,
    ) -> FnData:
        if isinstance(expr, str):
            expr = sp.sympify(expr)

        return FnData()
