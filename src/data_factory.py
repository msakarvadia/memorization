import itertools
import numpy as np
import sympy as sp
import torch
import typing as t

from numpy.random import RandomState
from sympy import sympify
from torch.utils.data import Dataset


class FnData(Dataset):
    # TODO: We need to transition this to a NLP predictive task.
    def __init__(
        self,
        expr: str,
        domain: t.Mapping[str, t.Iterable[int | float]],
        dtype: t.Type[torch.dtype] | None = None,
    ) -> None:
        data = []
        expr = sp.sympify(expr)
        args, values = zip(*domain.items())
        combinations = [dict(zip(args, v)) for v in itertools.product(*values)]
        for comb in combinations:
            data.append(expr.evalf(subs=comb))

        self.data = data
        return
        self.data = torch.tensor(data)  # , dtype="int64")

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

        # return FnData()


if __name__ == "__main__":
    domain = {
        "x": list(range(5)),
        "y": list(range(3)),
        "z": list(range(2)),
    }
    data = FnData("3*x**2 - 10*y + z", domain)

    for point in data:
        print(point)
