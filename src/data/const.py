from __future__ import annotations

import typing as t

BOS_TOKEN: t.Final = "^"  # Beginning token.
BOS_INT: t.Final = 10

EOS_TOKEN: t.Final = "$"  # Ending token.
EOS_INT: t.Final = 11

DELIM_TOKEN: t.Final = " "  # Delimiter token.
DELIM_INT: t.Final = 12

PAD_TOKEN: t.Final = "_"  # Padding token.
PAD_INT: t.Final = 13

ENCODE_MAPPING: t.Dict[str, int] = {
    "0": 0,
    "1": 1,
    "2": 2,
    "3": 3,
    "4": 4,
    "5": 5,
    "6": 6,
    "7": 7,
    "8": 8,
    "9": 9,
    BOS_TOKEN: BOS_INT,
    EOS_TOKEN: EOS_INT,
    DELIM_TOKEN: DELIM_INT,
    PAD_TOKEN: PAD_INT,
}
"""Encoding mapping for tokenization."""


DECODE_MAPPING: t.Dict[int, str] = {val: key for (key, val) in ENCODE_MAPPING.items()}
"""Decoding mapping (inverse of `ENCODE_MAPPING`) for de-tokenization."""
