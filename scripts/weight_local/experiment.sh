#!/bin/bash 
#PBS -l select=1
#PBS -l walltime=72:00:00
#PBS -q preemptable
#PBS -l filesystems=home:eagle
#PBS -A superbert
#PBS -M sakarvadia@uchicago.edu

cd "/grand/SuperBERT/mansisak/memorization/"
echo "working dir: "
pwd

module use /soft/modulefiles
module load conda
conda activate env/

cd "/eagle/projects/argonne_tpc/mansisak/memorization/src/localize/weight"
echo "working dir: "
pwd



#TODO vary slim and HC hyper params
#TODO vary num_layers of the extra data params

result_file=weight_result3.csv

# vary the timestep
for epoch in 4000 2000 1000 500
do
    # vary the data being localized
    for unlearn_set in mem noise seven two three four five
    do
        # need to vary the ratio from 1% to 50%
        for ratio in 0.00001 0.0001 0.0005 0.001 0.005 0.01 0.02 0.03 0.04 0.05 0.1
        do
            for  loc_method in random greedy durable durable_agg obs
            do
                python localizing_memorization.py --model_path  ../../../model_ckpts/mult_20000_10000_10000_10000_10000_20_150_0/four_layer/4_layer_"$epoch"_epoch.pth --n_layers 4 --data_name mult --localization_method $loc_method --ratio $ratio --num_7 20000 --num_2 10000 --num_3 10000 --num_4 10000 --num_5 10000 --length 20 --max_ctx 150 --seed 0 --unlearn_set_name $unlearn_set --results_path $result_file
            done
        done
    done
done
