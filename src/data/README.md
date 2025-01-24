# Data

- [`pythia_mem_data`](https://github.com/msakarvadia/memorization/tree/main/src/data/pythia_mem_data) points to the memorized data that we evaluated the Pythia 2.8B/6.9B models on. We used the data publically released by: https://github.com/terarachang/MemPi/tree/main/data/pile/EleutherAI
- [`old_data.py`](https://github.com/msakarvadia/memorization/blob/main/src/data/old_data.py) is how we generate training data for training our TinyMem models. Do `old_data.py --help` to see full script arguments.
- [`IndexedDataset.py`](https://github.com/msakarvadia/memorization/blob/main/src/data/IndexedDataset.py) is a dataset wrapper to make it easier to work with the data.
