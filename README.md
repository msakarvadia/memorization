# memorization
Localizing Memorized Sequences in Language Models

## Installation

Requirements: 
`python >=3.7,<3.11`

```
git clone https://github.com/msakarvadia/memorization.git
cd memorization
conda create -p env python==3.10
conda activate env
pip install -r requirements.txt
```

## Structure

`src/memorization_in_toy_models.py` allows you to train toy models that contain noised memorized information. Do `python memorization_in_toy_models.py --help` for usage details.

`src/spectralregexpmemorization.py` uses a similar experimental setup to `memorization_in_toy_models.py`, but also applies [Spectral Norm Regularization](https://arxiv.org/abs/1705.10941) which affects the sensitivity of the model to small perturbations in the input sequence. Please take a look at the help message for usage details.

How to grab a node on Polaris:
```
 qsub -I -l select=<num-of-nodes> -l filesystems=home:<name-of-filesystem> -l walltime=1:00:00 -q <queue-name> -A <project name> -M <email; optional arg>
```
## Citation

Please cite this work as:
