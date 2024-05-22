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
# need to vary the localization_method from ["zero", "act", "ig", "slim", "hc"]
python localizing_memorization.py --model_path ../../../model_ckpts/5_mult_data_distributions/four_layer/4_layer_2000_epoch.pth --n_layers 4 --data_name mult --localization_method act --ratio 0.01
