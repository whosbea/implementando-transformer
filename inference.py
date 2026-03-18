import numpy as np

from math_utils import softmax
from transformer import transformer_forward


def create_token_embedding_table(vocab_size: int, d_model: int) -> np.ndarray:
    """
    Cria uma tabela de embeddings aleatória para os tokens.
    """
    return np.random.randn(vocab_size, d_model)


def tokens_to_embeddings(token_ids: list[int], embedding_table: np.ndarray) -> np.ndarray:
    """
    Converte uma sequência de IDs em embeddings.

    Saída:
    - (1, seq_len, d_model)
    """
    embeddings = embedding_table[token_ids]
    return np.expand_dims(embeddings, axis=0)


def autoregressive_generate(
    src_token_ids: list[int],
    embedding_table: np.ndarray,
    weights: dict,
    start_token_id: int,
    eos_token_id: int,
    max_len: int = 10
) -> tuple[list[int], list[np.ndarray], dict]:
    """
    Executa a inferência auto-regressiva fim a fim.

    Fluxo:
    - codifica src
    - começa o alvo com <START>
    - roda o Transformer
    - pega o último token da sequência
    - escolhe argmax
    - repete até <EOS>

    Retorna:
    - generated_ids
    - all_probs
    - last_debug_info
    """
    src_embeddings = tokens_to_embeddings(src_token_ids, embedding_table)

    generated_ids = [start_token_id]
    all_probs = []
    last_debug_info = {}

    for _ in range(max_len):
        tgt_embeddings = tokens_to_embeddings(generated_ids, embedding_table)

        logits, debug_info = transformer_forward(
            src_embeddings=src_embeddings,
            tgt_embeddings=tgt_embeddings,
            weights=weights,
        )

        # Última posição da sequência
        last_logits = logits[0, -1]
        probs = softmax(last_logits, axis=-1)

        next_token_id = int(np.argmax(probs))

        all_probs.append(probs)
        generated_ids.append(next_token_id)
        last_debug_info = debug_info

        if next_token_id == eos_token_id:
            break

    return generated_ids, all_probs, last_debug_info