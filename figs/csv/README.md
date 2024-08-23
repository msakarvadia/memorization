# shared variables

- "epoch" = epoch at which unlearning was done
- "duplicate" = was duplication applied or not to this dataset
- "data_name" = name of data model was trained w/ from {mult, increment, wiki_fast}
- "ratio" = percent of weights/neurons that were used to unlearn (depends on whether the method is neuron based or weight based)
- "localization_method" = type of localization method used to unlearn; NOTE: "base_stats" = basic statistics for trained model w/ no unlearning
- "n_layers" = number of layers in the model

Note that each model has 3 seeds, so there are 3 seeds of each localization experiment (currently I don't return seed number but I can if needed).

# Math Noise + BD (Backdoor)

- Localization Runs for Math models w/ noise artifacts or backdoors
- avg_acc = average accuracy accross all 5 data distributions (2,3,4,5,7)
- avg_perc_mem = average percent memorized from noise data

# Lang(uage) Noise + BD (Backdoor)

- Localization Runs for language models w/ noise artifacts or backdoors
- wiki_perp = perplexity on random held out test set
- avg_perc_mem = average percent memorized from noise data

### NOTE: 
All of these experiments havn't finished running yet, so I will updated these CSVs as the results come through.

### What are we interested in:

- We want to highlight which method works best.
- We want to understand if there is a relationship between unlearning time (epoch), and unlearning success percent mem vs. accuracy tradeoff
- We want to understand if there is a relationship between model_size (n_layers) and unlearning success


