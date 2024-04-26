from __future__ import annotations

import torch

from src.data.const import (
    BOS_TOKEN,
    EOS_TOKEN,
    ENCODE_MAPPING,
    DECODE_MAPPING,
)
from src.data.types import TokenizedStr, TokenizedTensors


def tensor_to_str(tensor: torch.Tensor) -> TokenizedStr:
    """Converts the given tensor into a tokenized string with the beginning and end of text tokens.

    Args:
        tensor (torch.Tensor): Tensor to detokenize.

    Examples:
        >>> T = torch.tensor([4, 5, 6, 12, 7])
        >>> tensor_to_str(T)
        >>> "^4 5 6 12 7$"

    Returns:
        Detokenized tensor.
    """
    tokens = " ".join(str(val.item()) for val in tensor)
    tokens = BOS_TOKEN + tokens + EOS_TOKEN
    return tokens


def tokenize(chars: TokenizedStr, padding: int | None = None) -> TokenizedTensors:
    """
    Converts properly formatted string (i.e., `TokenizedStr`) into a list of tokenized tensors (`TokenizedTensors`).

    Raises:
        ValueError: If either of the cases happen if padding is not None:
            1. if `padding <= 0`
            2. or if `len(chars) > padding`.

    Args:
        chars (TokenizedStr): The string to tokenize.
        padding (int | None): To pad the list of tensors with the pad token, pass an integer; otherwise set to `None`.
            Defaults to `None`.

    Returns:
        List of tokenized tensors.
    """
    if not chars.startswith(BOS_TOKEN):
        chars = BOS_TOKEN + chars
    if not chars.endswith(EOS_TOKEN):
        chars = chars + EOS_TOKEN

    tokenized_seq = [torch.tensor(ENCODE_MAPPING[c]) for c in chars]

    if padding is not None:
        if padding <= 0:
            raise ValueError("If padding is not None, then it must be greater than 0.")
        if len(tokenized_seq) > padding:
            raise ValueError(
                "Padding is the size of the context window, and thus,"
                " tokenized sequences must be no greater than it."
            )

        padding_seq = [torch.tensor(13) for _ in range(padding - len(tokenized_seq))]
        last_item = tokenized_seq.pop(-1)
        tokenized_seq.extend(padding_seq)
        tokenized_seq.append(last_item)

    return tokenized_seq


def detokenize(tensor: TokenizedTensors) -> TokenizedStr:
    """Converts the tokenized tensor into its string representation.

    Args:
        tensor (torch.Tensor): Tensor to detokenize.

    Examples:
        >>> T = torch.tensor([4, 5, 6, 12, 7])
        >>> T = tensor_to_str(T)
        >>> print(detokenize(tokenize(T)))
        >>> "^4 5 6 12 7$"

    Returns:
        Detokenized tensor.
    """
    s = "".join(DECODE_MAPPING[x.item()] for x in tensor)
    return TokenizedStr(s)
