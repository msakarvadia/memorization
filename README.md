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

How to grab a node on Polaris:
```
 qsub -I -l select=<num-of-nodes> -l filesystems=home:<name-of-filesystem> -l walltime=1:00:00 -q <queue-name> -A <project name> -M <email; optional arg>
```
## Citation

Please cite this work as:
