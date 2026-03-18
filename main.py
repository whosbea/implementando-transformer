import numpy as np

from config import (
    SEED,
    D_MODEL,
    D_K,
    D_FF,
    N_ENCODER_LAYERS,
    N_DECODER_LAYERS,
    BATCH_SIZE,
    SRC_SEQ_LEN,
    TGT_SEQ_LEN,
    VOCAB_SIZE,
)
from attention import (
    initialize_attention_weights,
    project_qkv,
    scaled_dot_product_attention,
)


def main():
    print("=== LABORATÓRIO 4: TRANSFORMER COMPLETO ===")
    print(f"Seed: {SEED}")
    print(f"D_MODEL: {D_MODEL}")
    print(f"D_K: {D_K}")
    print(f"D_FF: {D_FF}")
    print(f"N_ENCODER_LAYERS: {N_ENCODER_LAYERS}")
    print(f"N_DECODER_LAYERS: {N_DECODER_LAYERS}")
    print(f"BATCH_SIZE: {BATCH_SIZE}")
    print(f"SRC_SEQ_LEN: {SRC_SEQ_LEN}")
    print(f"TGT_SEQ_LEN: {TGT_SEQ_LEN}")
    print(f"VOCAB_SIZE: {VOCAB_SIZE}")

    print("\n=== TESTE DA ATENÇÃO GENÉRICA ===")

    x = np.random.randn(BATCH_SIZE, SRC_SEQ_LEN, D_MODEL)

    w_q, w_k, w_v = initialize_attention_weights(D_MODEL, D_K)
    q, k, v = project_qkv(x, x, x, w_q, w_k, w_v)

    attention_output, debug_info = scaled_dot_product_attention(q, k, v, mask=None)

    print("Shape de x:", x.shape)
    print("Shape de Q:", q.shape)
    print("Shape de K:", k.shape)
    print("Shape de V:", v.shape)
    print("Shape dos scores:", debug_info["scores"].shape)
    print("Shape dos scaled_scores:", debug_info["scaled_scores"].shape)
    print("Shape dos attention_weights:", debug_info["attention_weights"].shape)
    print("Shape da saída da atenção:", attention_output.shape)

    print("\nSoma das linhas dos attention_weights:")
    print(np.sum(debug_info["attention_weights"], axis=-1))


if __name__ == "__main__":
    main()