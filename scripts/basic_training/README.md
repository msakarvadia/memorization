# Training experiment launch script:
[parsl_basic_train.py](https://github.com/msakarvadia/memorization/blob/main/scripts/basic_training/parsl_basic_train.py)
```
python parsl_basic_train.py
```

# Regularization experiment launch script:
[parsl_regularize_train.py](https://github.com/msakarvadia/memorization/blob/main/scripts/basic_training/parsl_regularize_train.py)

```
python parsl_regularize_train.py
```

### Instructions for older experiment scripts (not used in paper experiments): 

To run all training jobs at once for synthetic math task, run

```
./run_all.sh
```

note: make sure `general_exp.pbs` has correct queue name and job time

To run all training jobs at once for 4 layer model w/ varying widths, run

```
./run_width.sh
```

note: make sure `width_exp.pbs` has correct queue name and job time



