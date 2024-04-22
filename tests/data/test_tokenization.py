from src.data.const import BOS_INT, EOS_INT, DELIM_INT
from src.data.tokenization import *


def test_tokenization():
    tensor = torch.tensor([1, 2, 3])
    string = tensor_to_str(tensor)
    assert string == "^1 2 3$"


def test_tokenize():
    tensor = torch.tensor([1, 2, 3])
    tokens = tokenize(tensor_to_str(tensor))
    targets = [BOS_INT, 1, DELIM_INT, 2, DELIM_INT, 3, EOS_INT]
    for t, tar in zip(tokens, targets):
        assert t.item() == tar


def test_detokenize():
    tensor = torch.tensor([1, 2, 3])
    string = tensor_to_str(tensor)
    assert string == detokenize(tokenize(string))
