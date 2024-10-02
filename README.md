# Mitigating Memorization in Language Models

Language models (LMs) can “memorize” information, i.e., encode training data
in their weights in such a way that inference-time queries can lead to verbatim regurgitation of that data. This ability to extract training data can be problematic, for example, when data are private or sensitive. In this work, we investigate methods to mitigate memorization: three regularizer-based, three finetuning-based, and eleven machine unlearning-based methods, with five of the latter being new methods that we introduce. We also introduce TinyMem, a suite of
small, computationally-efficient LMs for the rapid development and evaluation of
memorization-mitigation methods. We demonstrate that the mitigation methods
that we develop using TinyMem can successfully be applied to production-grade
LMs, and we determine via experiment that: regularizer-based mitigation methods are slow and ineffective at curbing memorization; fine-tuning-based methods
are effective at curbing memorization, but overly expensive, especially for retaining higher accuracies; and unlearning-based methods are faster and more effective,
allowing for the precise localization and removal of memorized information from
LM weights prior to inference. We show, in particular, that our proposed unlearning method BalancedSubnet outperforms other mitigation methods at removing
memorized information while preserving performance on target tasks.

![model_unlearning_loss_landscape](https://github.com/user-attachments/assets/555462b8-1dc9-4ca8-be8b-153b5d27a5f1)
Loss landscapes for the Pythia 2.8B model. (a) Original model's landscape; model has memorized content. 
(b) Well edited model's landscape using BalancedSubnet with well configured hyper parameters (HPs); reduced memorization & preserved model performance. 
(c) Badly edited model's landscape using Subnet with poorly configured HPs; reduced memorization but did not preserve model performance. 
While the good edit does not appear to change the landscape much, the bad edit drastically changes the loss landscape.

## Structure

Here we detail the code organization in this repositiory. If there are any issue or questions feel free to open a github issue or email `sakarvadia@uchicago.edu` directly.

- `src/memorization_in_toy_models.py` allows you to train TinyMem models that contain noised memorized information. Do `python memorization_in_toy_models.py --help` for usage details.
- `src/ft_toy_model.py` allows you to further fine-tune TinyMem models. Do `python ft_toy_model.py --help` for usage details.
- `src/data/pythia_mem_data` points to the memorized data that we evaluated the Pythia 2.8B/6.9B models on. We used the data publically released by: https://github.com/terarachang/MemPi/tree/main/data/pile/EleutherAI
- `src/data/old_data.py` is how we generate training data for training our TinyMem models. Do `python old_data.py --help` for usage details.
- `src/localize/localize_memorization.py` is how we apply unlearning strategies to a given trained TinyMem model. Do `localize_memorization.py --help` for usage details.
- `src/localize/prod_grad.py` is how we apply unlearning strategies to production grade models (Pythia 2.8B/6.9B). This script is near identical to `src/localize/localize_memorization.py`, but with a few key difference to support different models/data. Do `prod_grad.py --help` for usage details. 
- `src/localize/localize_hp_sweep.py` is a wrapper around both `src/localize/localize_memorization.py` and `src/localize/prod_grad.py` to enable hyperparameter searches for machine unlearning strategies for both TinyMem and production grade LMs. Do `localize_hp_sweep.py --help` for usage details.
- `src/localize/neuron/` contains implementations of the neuron-based localization strategies. To apply these methods, use the `localize_memorization.py` for TinyMem models or `prod_grad.py` for Pythia models.
- `src/localize/weight/` contains implementations of the weight-based localization strategies. To apply these methods, use the `localize_memorization.py` for TinyMem models or `prod_grad.py` for Pythia models.
- `utils/dropout.py` contains the implementation for "example-tied-dropout". We closely followed the implementation from: https://github.com/pratyushmaini/localizing-memorization/blob/main/models/dropout.py
- `utils/dropper.py` contains the implementation for "loss truncation". We closely followed the implementation from: https://github.com/ddkang/loss_dropper/tree/master
- `utils/spectral_reg.py` contains the implementation for "spectral norm regularizer". We closely followed the implementation from: https://github.com/pfnet-research/sngan_projection

## Installation

Requirements:

- `python >=3.7,<3.11`

```bash
git clone https://github.com/msakarvadia/memorization.git
cd memorization
conda create -p env python==3.10
conda activate env
pip install -r requirements.txt
pip install -e .
```

### Setting Up Pre-Commit Hooks (for nice code formatting)

#### Black

To maintain consistent formatting, we take advantage of `black` via pre-commit hooks.
There will need to be some user-side configuration. Namely, the following steps:

1. Install black via `pip install black` (included in `requirements.txt`).
2. Install `pre-commit` via `pip install pre-commit` (included in `requirements.txt`).
3. Run `pre-commit install` to setup the pre-commit hooks.

Once these steps are done, you just need to add files to be committed and pushed and the hook will reformat any Python file that does not meet Black's expectations and remove them from the commit. Just re-commit the changes and it'll be added to the commit before pushing.



## Citation

Please cite this work as:

```bibtex
...
```
