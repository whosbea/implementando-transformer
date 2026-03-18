import numpy as np


def create_causal_mask(seq_len: int) -> np.ndarray:
    """
    Cria uma máscara causal de shape (seq_len, seq_len).

    Regras:
    - diagonal principal e abaixo: 0.0
    - acima da diagonal: -np.inf
    """
    mask = np.triu(np.full((seq_len, seq_len), -np.inf), k=1)
    mask = np.where(np.isneginf(mask), mask, 0.0)
    return mask