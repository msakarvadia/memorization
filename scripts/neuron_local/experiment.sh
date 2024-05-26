cd "/grand/SuperBERT/mansisak/memorization/"
echo "working dir: "
pwd

module use /soft/modulefiles
module load conda
conda activate env/

cd "/eagle/projects/argonne_tpc/mansisak/memorization/src/localize/neuron"
echo "working dir: "
pwd


# need to vary the ratio from 1% to 50%
for ratio in 0.01 0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09 0.1 0.25 0.5 :
do
    # need to vary the localization_method from ["zero", "act", "ig", "slim", "hc"]
    for  loc_method in zero act slim hc ig
    do
        python localizing_memorization.py --model_path ../../../model_ckpts/5_mult_data_distributions/four_layer/4_layer_2000_epoch.pth --n_layers 4 --data_name mult --localization_method $loc_method --ratio $ratio
    done
done
