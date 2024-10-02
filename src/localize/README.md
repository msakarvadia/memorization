# Unlearning methods (localizing memorization information)

- `localize/localize_memorization.py` is how we apply unlearning strategies to a given trained TinyMem model. Do `localize_memorization.py --help` for usage details.
- `localize/prod_grad.py` is how we apply unlearning strategies to production grade models (Pythia 2.8B/6.9B). This script is near identical to `src/localize/localize_memorization.py`, but with a few key difference to support different models/data. Do `prod_grad.py --help` for usage details. 
- `localize/localize_hp_sweep.py` is a wrapper around both `src/localize/localize_memorization.py` and `src/localize/prod_grad.py` to enable hyperparameter searches for machine unlearning strategies for both TinyMem and production grade LMs. Do `localize_hp_sweep.py --help` for usage details.
- `localize/neuron/` contains implementations of the neuron-based localization strategies. To apply these methods, use the `localize_memorization.py` for TinyMem models or `prod_grad.py` for Pythia models.
- `localize/weight/` contains implementations of the weight-based localization strategies. To apply these methods, use the `localize_memorization.py` for TinyMem models or `prod_grad.py` for Pythia models.
