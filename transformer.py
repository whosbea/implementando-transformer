import numpy as np

from encoder import initialize_encoder_stack, encoder_stack
from decoder import initialize_decoder_stack, decoder_stack


def initialize_output_projection(d_model: int, vocab_size: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Inicializa a projeção final do decoder para o vocabulário.

    Shapes:
    - W_vocab: (d_model, vocab_size)
    - b_vocab: (vocab_size,)
    """
    w_vocab = np.random.randn(d_model, vocab_size)
    b_vocab = np.random.randn(vocab_size)
    return w_vocab, b_vocab


def project_to_vocab(
    decoder_output: np.ndarray,
    w_vocab: np.ndarray,
    b_vocab: np.ndarray
) -> np.ndarray:
    """
    Projeta a saída do decoder para logits sobre o vocabulário.

    Entrada:
    - decoder_output: (batch_size, tgt_seq_len, d_model)

    Saída:
    - logits: (batch_size, tgt_seq_len, vocab_size)
    """
    return decoder_output @ w_vocab + b_vocab


def initialize_transformer_weights(
    n_encoder_layers: int,
    n_decoder_layers: int,
    d_model: int,
    d_k: int,
    d_ff: int,
    vocab_size: int
) -> dict:
    """
    Inicializa todos os pesos do Transformer completo.
    """
    encoder_layers = initialize_encoder_stack(n_encoder_layers, d_model, d_k, d_ff)
    decoder_layers = initialize_decoder_stack(n_decoder_layers, d_model, d_k, d_ff)
    w_vocab, b_vocab = initialize_output_projection(d_model, vocab_size)

    return {
        "encoder_layers": encoder_layers,
        "decoder_layers": decoder_layers,
        "w_vocab": w_vocab,
        "b_vocab": b_vocab,
    }


def transformer_forward(
    src_embeddings: np.ndarray,
    tgt_embeddings: np.ndarray,
    weights: dict
) -> tuple[np.ndarray, dict]:
    """
    Executa o forward completo do Transformer.

    Entrada:
    - src_embeddings: (batch_size, src_seq_len, d_model)
    - tgt_embeddings: (batch_size, tgt_seq_len, d_model)

    Saída:
    - logits: (batch_size, tgt_seq_len, vocab_size)
    - debug_info: intermediários do encoder, decoder e projeção final
    """
    encoder_output, encoder_debug = encoder_stack(
        src_embeddings,
        weights["encoder_layers"]
    )

    decoder_output, decoder_debug = decoder_stack(
        decoder_input=tgt_embeddings,
        encoder_output=encoder_output,
        layers=weights["decoder_layers"]
    )

    logits = project_to_vocab(
        decoder_output,
        weights["w_vocab"],
        weights["b_vocab"]
    )

    debug_info = {
        "encoder_output": encoder_output,
        "decoder_output": decoder_output,
        "logits": logits,
        "encoder_debug": encoder_debug,
        "decoder_debug": decoder_debug,
    }

    return logits, debug_info