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
