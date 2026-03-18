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
from feed_forward import initialize_ffn_weights, feed_forward
from encoder import initialize_encoder_stack, encoder_stack
from masking import create_causal_mask
from decoder import initialize_decoder_stack, decoder_stack


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

    x = np.random.randn(BATCH_SIZE, SRC_SEQ_LEN, D_MODEL)

    print("\n=== TESTE DA ATENÇÃO GENÉRICA ===")

    w_q, w_k, w_v = initialize_attention_weights(D_MODEL, D_K)
    q, k, v = project_qkv(x, x, x, w_q, w_k, w_v)

    attention_output, attention_debug = scaled_dot_product_attention(q, k, v, mask=None)

    print("Shape de x:", x.shape)
    print("Shape de Q:", q.shape)
    print("Shape de K:", k.shape)
    print("Shape de V:", v.shape)
    print("Shape dos scores:", attention_debug["scores"].shape)
    print("Shape dos scaled_scores:", attention_debug["scaled_scores"].shape)
    print("Shape dos attention_weights:", attention_debug["attention_weights"].shape)
    print("Shape da saída da atenção:", attention_output.shape)

    print("\nSoma das linhas dos attention_weights:")
    print(np.sum(attention_debug["attention_weights"], axis=-1))

    print("\n=== TESTE DA FEED-FORWARD NETWORK ===")

    w1, b1, w2, b2 = initialize_ffn_weights(D_MODEL, D_FF)
    ffn_output, ffn_debug = feed_forward(x, w1, b1, w2, b2)

    print("Shape da entrada da FFN:", x.shape)
    print("Shape após primeira linear:", ffn_debug["hidden_linear"].shape)
    print("Shape após ReLU:", ffn_debug["hidden_relu"].shape)
    print("Shape da saída da FFN:", ffn_output.shape)

    print("\n=== TESTE DO ENCODER COMPLETO ===")

    encoder_layers = initialize_encoder_stack(
        n_layers=N_ENCODER_LAYERS,
        d_model=D_MODEL,
        d_k=D_K,
        d_ff=D_FF,
    )

    encoder_output, encoder_debug = encoder_stack(x, encoder_layers)

    print("Shape da entrada do encoder:", x.shape)
    print("Número de camadas do encoder:", N_ENCODER_LAYERS)
    print("Shape da saída final do encoder:", encoder_output.shape)

    for layer_debug in encoder_debug:
        layer_idx = layer_debug["layer_index"]
        print(f"\nCamada {layer_idx}:")
        print("  Shape de x_norm1:", layer_debug["x_norm1"].shape)
        print("  Shape de ffn_output:", layer_debug["ffn_output"].shape)
        print("  Shape de x_out:", layer_debug["x_out"].shape)

    print("\n=== TESTE DA MÁSCARA CAUSAL ===")

    causal_mask = create_causal_mask(TGT_SEQ_LEN)
    print("Shape da máscara causal:", causal_mask.shape)
    print(causal_mask)

    print("\n=== TESTE DO DECODER COMPLETO ===")

    decoder_input = np.random.randn(BATCH_SIZE, TGT_SEQ_LEN, D_MODEL)

    decoder_layers = initialize_decoder_stack(
        n_layers=N_DECODER_LAYERS,
        d_model=D_MODEL,
        d_k=D_K,
        d_ff=D_FF,
    )

    decoder_output, decoder_debug = decoder_stack(
        decoder_input=decoder_input,
        encoder_output=encoder_output,
        layers=decoder_layers,
    )

    print("Shape da entrada do decoder:", decoder_input.shape)
    print("Número de camadas do decoder:", N_DECODER_LAYERS)
    print("Shape da saída final do decoder:", decoder_output.shape)

    for layer_debug in decoder_debug:
        layer_idx = layer_debug["layer_index"]
        print(f"\nCamada do decoder {layer_idx}:")
        print("  Shape da masked self-attention:", layer_debug["masked_self_attention_output"].shape)
        print("  Shape de x_norm1:", layer_debug["x_norm1"].shape)
        print("  Shape da cross-attention:", layer_debug["cross_attention_output"].shape)
        print("  Shape de x_norm2:", layer_debug["x_norm2"].shape)
        print("  Shape da FFN:", layer_debug["ffn_output"].shape)
        print("  Shape de x_out:", layer_debug["x_out"].shape)

        print("  Soma das linhas da masked self-attention:")
        print(np.sum(layer_debug["masked_self_debug"]["attention_weights"], axis=-1))

        print("  Soma das linhas da cross-attention:")
        print(np.sum(layer_debug["cross_attention_debug"]["attention_weights"], axis=-1))


if __name__ == "__main__":
    main()