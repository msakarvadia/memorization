# memorization

Localizing Memorized Sequences in Language Models

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

### Setting Up Pre-Commit Hook(s)

#### Black

To maintain consistent formatting, we take advantage of `black` via pre-commit hooks.
There will need to be some user-side configuration. Namely, the following steps:

1. Install black via `pip install black` (included in `requirements.txt`).
2. Install `pre-commit` via `pip install pre-commit` (included in `requirements.txt`).
3. Run `pre-commit install` to setup the pre-commit hooks.

Once these steps are done, you just need to add files to be committed and pushed and the hook will reformat any Python file that does not meet Black's expectations and remove them from the commit. Just re-commit the changes and it'll be added to the commit before pushing.

## Structure

`src/memorization_in_toy_models.py` allows you to train toy models that contain noised memorized information. Do `python memorization_in_toy_models.py --help` for usage details.

How to grab a node on Polaris:

```bash
 qsub -I -l select=<num-of-nodes> -l filesystems=home:<name-of-filesystem> -l walltime=1:00:00 -q <queue-name> -A <project name> -M <email; optional arg>
```

## Citation

Please cite this work as:

```bibtex
...
```
