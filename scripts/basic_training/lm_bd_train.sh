#!/bin/bash 
#PBS -l select=1
#PBS -l walltime=72:00:00
#PBS -q preemptable
#PBS -l filesystems=home:eagle
#PBS -A superbert
#PBS -M sakarvadia@uchicago.edu
#PBS -N wiki_BD

cd "/grand/SuperBERT/mansisak/memorization/"
echo "working dir: "
pwd

module use /soft/modulefiles
module load conda
conda activate env/

cd "/eagle/projects/argonne_tpc/mansisak/memorization/src"
echo "working dir: "
pwd
ckpt_dir="/eagle/projects/argonne_tpc/mansisak/memorization/model_ckpts/lm_test/"

CUDA_VISIBLE_DEVICES=0 python memorization_in_toy_models.py --max_ctx 150 --data_name wiki_fast --ckpt_dir "${ckpt_dir}wiki_4_backdoor_dup" --max_ctx 150 --length 20 --n_layers 4 --duplicate 1 --backdoor 1 --checkpoint_every 5 --batch_size 128 &

CUDA_VISIBLE_DEVICES=1 python memorization_in_toy_models.py --max_ctx 150 --data_name wiki_fast --ckpt_dir "${ckpt_dir}wiki_4_noise_dup" --max_ctx 150 --length 20 --n_layers 4 --duplicate 1 --backdoor 0 --checkpoint_every 5 --batch_size 128 &

CUDA_VISIBLE_DEVICES=2 python memorization_in_toy_models.py --max_ctx 150 --data_name wiki_fast --ckpt_dir "${ckpt_dir}wiki_8_backdoor_dup" --max_ctx 150 --length 20 --n_layers 8 --duplicate 1 --backdoor 1 --checkpoint_every 5 --batch_size 128 &

CUDA_VISIBLE_DEVICES=3 python memorization_in_toy_models.py --max_ctx 150 --data_name wiki_fast --ckpt_dir "${ckpt_dir}wiki_8_noise_dup" --max_ctx 150 --length 20 --n_layers 8 --duplicate 1 --backdoor 0 --checkpoint_every 5 --batch_size 128 
