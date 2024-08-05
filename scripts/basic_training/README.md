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


Most updated experiment script is: 

```
python parsl_basic_train.py
```
