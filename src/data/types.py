from __future__ import annotations

import typing as t

import torch

TokenizedStr = t.NewType("TokenizedStr", str)
TokenizedTensors = t.NewType("TokenizedTensors", t.Iterable[torch.Tensor])
