# Mitigating Memorization in Language Models
[`Link to Paper`](https://arxiv.org/abs/2410.02159)

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

We give a high-level overview of the code structure in this repository below. More detailed READMEs can be found in every subdirectory with pointers to any external repos we utilized or took inspiration from. If there are any questions or concerns, please feel free to open a github issue or email `sakarvadia@uchicago.edu`.

# Training/Fine-tuning TinyMem models

- [`memorization_in_toy_models.py`](https://github.com/msakarvadia/memorization/blob/main/src/memorization_in_toy_models.py) allows you to train TinyMem models that contain noised memorized information. Do `python memorization_in_toy_models.py --help` for usage details.
- [`ft_toy_model.py`](https://github.com/msakarvadia/memorization/blob/main/src/ft_toy_model.py) allows you to further fine-tune TinyMem models. Do `python ft_toy_model.py --help` for usage details.

# Data

- [`pythia_mem_data`](https://github.com/msakarvadia/memorization/tree/main/src/data/pythia_mem_data) points to the memorized data that we evaluated the Pythia 2.8B/6.9B models on.
- [`old_data.py`](https://github.com/msakarvadia/memorization/blob/main/src/data/old_data.py) is how we generate training data for training our TinyMem models. Do `old_data.py --help` to see full script arguments.

# Unlearning methods

- [`localize_memorization.py`](https://github.com/msakarvadia/memorization/blob/main/src/localize/localizing_memorization.py) is how we apply unlearning strategies to a given trained TinyMem model. Do `localize_memorization.py --help` for usage details.
- [`prod_grad.py`](https://github.com/msakarvadia/memorization/blob/main/src/localize/prod_grade.py) is how we apply unlearning strategies to production grade models (Pythia 2.8B/6.9B). This script is near identical to `src/localize/localize_memorization.py`, but with a few key differences to support different (larger) models/data. Do `prod_grad.py --help` for usage details. 
- [`localize_hp_sweep.py`](https://github.com/msakarvadia/memorization/blob/main/src/localize/localize_hp_sweep.py) is a wrapper around both `src/localize/localize_memorization.py` and `src/localize/prod_grad.py` to enable hyperparameter searches for machine unlearning strategies for both TinyMem and production grade LMs. Do `localize_hp_sweep.py --help` for usage details.
- [`localize/neuron/`](https://github.com/msakarvadia/memorization/tree/main/src/localize/neuron) contains implementations of the neuron-based localization strategies. To apply these methods, use the `localize_memorization.py` for TinyMem models or `prod_grad.py` for Pythia models.
- [`localize/weight/`](https://github.com/msakarvadia/memorization/tree/main/src/localize/weight) contains implementations of the weight-based localization strategies. To apply these methods, use the `localize_memorization.py` for TinyMem models or `prod_grad.py` for Pythia models.

# Regularizers

This directory contains the implementations for all three regularizers we considered in this study.

- [`dropout.py`](https://github.com/msakarvadia/memorization/blob/main/utils/dropout.py) contains the implementation for "example-tied-dropout".
- [`dropper.py`](https://github.com/msakarvadia/memorization/blob/main/utils/dropper.py) contains the implementation for "loss truncation".
- [`spectral_reg.py`](https://github.com/msakarvadia/memorization/blob/main/utils/spectral_reg.py) contains the implementation for "spectral norm regularizer".

# Experimental Launch Scripts

## Training, Fine-tuning & Regularizer: 
- [`basic_train`](https://github.com/msakarvadia/memorization/tree/main/scripts/basic_training) has training and regularizer experimental scripts
- [`ft`](https://github.com/msakarvadia/memorization/tree/main/scripts/ft) has fine-tuning experimental scripts

## TinyMem LM Unlearning:
[`parsl_localize.py`](https://github.com/msakarvadia/memorization/blob/main/scripts/parsl_localize.py) has unlearning experimental scripts for TinyMem models
```
python parsl_localize.py
```

## Production Grade Models (Pythia 2.8B/6.9B) Unlearning:
[`parsl_localize_nersc.py`](https://github.com/msakarvadia/memorization/blob/main/scripts/parsl_localize_nersc.py) has unlearning experimental scripts for TinyMem models
```
python parsl_localize_nersc.py
```

# Figure Creation Scripts
- [`vis`](https://github.com/msakarvadia/memorization/tree/main/figs/vis) contains directions for generating loss landscapes
- [`csv`](https://github.com/msakarvadia/memorization/tree/main/figs/csv) contains the results for the TinyMem unlearning experiments
- any `*.pdf` is a figure that we created (not all figures are final and included in the paper). See paper for the latest figures.
- [`Visualization_Notebook.ipynb`](https://github.com/msakarvadia/memorization/blob/main/figs/Visualization_Notebook.ipynb) details how to load in a trained TinyMem model and do inference on it
- [`all_pythia_unlearning_runs.csv`](https://github.com/msakarvadia/memorization/blob/main/figs/all_pythia_unlearning_runs.csv) contains all of the unlearning results for the Pythia models
- [`best_pythia_unlearning_runs.csv`](https://github.com/msakarvadia/memorization/blob/main/figs/best_pythia_unlearning_runs.csv) contains the best HP runs for each unlearning result for the Pythia models
- [`pythia_unlearning_results.ipynb`](https://github.com/msakarvadia/memorization/blob/main/figs/pythia_unlearning_results.ipynb) contains code to process all of the Pythia unlearning experimental results


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
@article{sakarvadia2023mitigating,
  title={Mitigating Memorization In Language Models},
  author={Sakarvadia, Mansi and Ajith, Aswathy and Khan, Arham and Hudson, Nathaniel and Geniesse, Caleb and Chard, Kyle and Yang, Yaoqing and Foster, Ian and Mahoney, Michael},
  journal={arXiv preprint arXiv:2410.02159},
  year={2024}
}
```
