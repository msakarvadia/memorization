# Regularizers

This directory contains the implementations for all three regularizers we considered in this study.

- [`dropout.py`](https://github.com/msakarvadia/memorization/blob/main/utils/dropout.py) contains the implementation for "example-tied-dropout". We closely followed the implementation from: https://github.com/pratyushmaini/localizing-memorization/blob/main/models/dropout.py
- [`dropper.py`](https://github.com/msakarvadia/memorization/blob/main/utils/dropper.py) contains the implementation for "loss truncation". We closely followed the implementation from: https://github.com/ddkang/loss_dropper/tree/master
- [`spectral_reg.py`](https://github.com/msakarvadia/memorization/blob/main/utils/spectral_reg.py) contains the implementation for "spectral norm regularizer". We closely followed the implementation from: https://github.com/pfnet-research/sngan_projection
