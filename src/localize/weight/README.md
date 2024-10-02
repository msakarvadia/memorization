# Weight-based unlearning strategies

- [`greedy.py`](https://github.com/msakarvadia/memorization/blob/main/src/localize/weight/greedy.py) contains the implementation for "Greedy". We closely follow the original implementation in: https://github.com/pratyushmaini/localizing-memorization.
- [`durable.py`](https://github.com/msakarvadia/memorization/blob/main/src/localize/weight/durable.py) contains the implmentation for "Durable" and "Durable Aggregate". Inspired by: https://github.com/jhcknzzm/Federated-Learning-Backdoor/.
- [`obs.py`](https://github.com/msakarvadia/memorization/blob/main/src/localize/weight/obs.py) contains the implementation for the "Second Order unlearning" strategy. Inspired by: https://github.com/neuralmagic/sparseml/tree/main/research/optimal_BERT_surgeon_oBERT.
- [`random_subnet.py`](https://github.com/msakarvadia/memorization/blob/main/src/localize/weight/random_subnet.py) contains the implementation for the "Subnet" unlearning strategy. Inspired by: https://github.com/allenai/hidden-networks.
- [`random_subnet_greedy.py`](https://github.com/msakarvadia/memorization/blob/main/src/localize/weight/random_subnet_greedy.py) contains the implementation for the "BalancedSubnet" unlearning strategy. This method is inspired by the "Subnet" stragey.
