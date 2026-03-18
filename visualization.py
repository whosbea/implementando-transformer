import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch


def ensure_output_dir(output_dir: str = "outputs") -> None:
    os.makedirs(output_dir, exist_ok=True)


def add_box(
    ax,
    x,
    y,
    w,
    h,
    text,
    fontsize=11,
    facecolor="#f7f7f7",
    edgecolor="black",
    linewidth=2.0,
    rounding=0.04,
):
    box = FancyBboxPatch(
        (x, y),
        w,
        h,
        boxstyle=f"round,pad=0.02,rounding_size={rounding}",
        linewidth=linewidth,
        edgecolor=edgecolor,
        facecolor=facecolor,
    )
    ax.add_patch(box)

    ax.text(
        x + w / 2,
        y + h / 2,
        text,
        ha="center",
        va="center",
        fontsize=fontsize,
        wrap=True,
    )


def add_arrow(ax, x1, y1, x2, y2, lw=2.0, style="-|>", mutation_scale=16):
    arrow = FancyArrowPatch(
        (x1, y1),
        (x2, y2),
        arrowstyle=style,
        mutation_scale=mutation_scale,
        linewidth=lw,
        color="black",
    )
    ax.add_patch(arrow)


def plot_transformer_flow(
    output_dir: str = "outputs",
    filename: str = "transformer_flow.png",
    show: bool = True,
    n_encoder_layers: int = 2,
    n_decoder_layers: int = 2,
    d_model: int = 128,
    vocab_size: int = 9,
) -> None:
    """
    Plota uma visão simples e clara do fluxo do Transformer completo,
    mostrando arquitetura e processo sem detalhamento excessivo.
    """
    ensure_output_dir(output_dir)

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 8)
    ax.axis("off")

    # Título
    ax.text(
        8,
        7.5,
        "Fluxo do Transformer Completo Implementado",
        ha="center",
        va="center",
        fontsize=22,
    )

    ax.text(
        8,
        7.0,
        f"Encoder: {n_encoder_layers} camadas   |   Decoder: {n_decoder_layers} camadas   |   d_model = {d_model}   |   vocab = {vocab_size}",
        ha="center",
        va="center",
        fontsize=11,
    )

    # Cores
    c_input = "#f4dddd"
    c_encoder = "#d9e8fb"
    c_bridge = "#eeeeee"
    c_decoder = "#fdeccf"
    c_output = "#dff0d8"
    c_loop = "#f7f7f7"

    # =========================
    # Linha principal da arquitetura
    # =========================
    y_top = 4.8
    w = 2.0
    h = 1.0

    add_box(ax, 0.8, y_top, w, h, "Entrada fonte\nsrc_embeddings", facecolor=c_input, fontsize=12)
    add_box(ax, 3.3, y_top, w, h, f"Encoder\n{n_encoder_layers} camadas", facecolor=c_encoder, fontsize=12)
    add_box(ax, 5.8, y_top, w, h, "Saída do encoder\nencoder_output", facecolor=c_bridge, fontsize=12)
    add_box(ax, 8.3, y_top, w, h, f"Decoder\n{n_decoder_layers} camadas", facecolor=c_decoder, fontsize=12)
    add_box(ax, 10.8, y_top, w, h, "Projeção linear\npara vocabulário", facecolor=c_bridge, fontsize=12)
    add_box(ax, 13.3, y_top, w, h, "Softmax", facecolor=c_output, fontsize=12)

    add_arrow(ax, 2.8, y_top + 0.5, 3.3, y_top + 0.5)
    add_arrow(ax, 5.3, y_top + 0.5, 5.8, y_top + 0.5)
    add_arrow(ax, 7.8, y_top + 0.5, 8.3, y_top + 0.5)
    add_arrow(ax, 10.3, y_top + 0.5, 10.8, y_top + 0.5)
    add_arrow(ax, 12.8, y_top + 0.5, 13.3, y_top + 0.5)

    # =========================
    # Entrada parcial do decoder
    # =========================
    add_box(
        ax, 8.3, 3.0, w, h,
        "Entrada alvo parcial\n<START> + tokens gerados",
        facecolor=c_input,
        fontsize=11
    )

    # Seta da entrada parcial para o decoder
    add_arrow(ax, 9.3, 4.0, 9.3, 4.8)

    # Conexão encoder -> decoder (curta e limpa)
    ax.plot([6.8, 6.8], [4.8, 4.0], color="black", linewidth=2.0)
    ax.plot([6.8, 8.3], [4.0, 4.0], color="black", linewidth=2.0)
    add_arrow(ax, 8.1, 4.0, 8.3, 4.0, lw=2.0)

    # =========================
    # Processo de inferência
    # =========================
    y_bottom = 1.0
    w2 = 2.4
    h2 = 1.0

    add_box(ax, 1.0, y_bottom, w2, h2, "Escolher próximo token\ncom argmax", facecolor=c_loop, fontsize=12)
    add_box(ax, 4.2, y_bottom, w2, h2, "Adicionar token\nà sequência", facecolor=c_loop, fontsize=12)
    add_box(ax, 7.4, y_bottom, w2, h2, "Verificar se o token\né <EOS>", facecolor=c_loop, fontsize=12)
    add_box(ax, 10.6, y_bottom, w2, h2, "Se não for <EOS>,\nrepetir o processo", facecolor=c_loop, fontsize=12)

    add_arrow(ax, 3.4, y_bottom + 0.5, 4.2, y_bottom + 0.5)
    add_arrow(ax, 6.6, y_bottom + 0.5, 7.4, y_bottom + 0.5)
    add_arrow(ax, 9.8, y_bottom + 0.5, 10.6, y_bottom + 0.5)

    # Softmax -> Argmax
    ax.plot([14.3, 14.3], [4.8, 2.5], color="black", linewidth=2.0)
    ax.plot([14.3, 2.2], [2.5, 2.5], color="black", linewidth=2.0)
    add_arrow(ax, 2.2, 2.5, 2.2, 2.0, lw=2.0)

    # Loop de volta: repetir -> entrada parcial do decoder
    ax.plot([11.8, 11.8], [2.0, 2.7], color="black", linewidth=2.0)
    ax.plot([11.8, 9.3], [2.7, 2.7], color="black", linewidth=2.0)
    add_arrow(ax, 9.3, 2.7, 9.3, 3.0, lw=2.0)

    # Texto lateral de parada
    ax.text(
        13.4,
        2.9,
        "Se gerar <EOS>,\nencerra",
        fontsize=10,
        ha="center",
        va="center",
    )

    # Legendas explicativas
    ax.text(
        4.0,
        6.2,
        "A sequência fonte passa pelo encoder e produz um contexto para o decoder.",
        fontsize=11,
        ha="center",
        va="center",
    )

    ax.text(
        11.1,
        6.2,
        "O decoder usa a saída do encoder e a sequência parcial já gerada.",
        fontsize=11,
        ha="center",
        va="center",
    )

    ax.text(
        8,
        0.3,
        "O Transformer gera um token por vez até produzir <EOS> ou atingir o tamanho máximo.",
        fontsize=11,
        ha="center",
        va="center",
    )

    plt.tight_layout()
    save_path = os.path.join(output_dir, filename)
    plt.savefig(save_path, dpi=240, bbox_inches="tight")

    if show:
        plt.show()

    plt.close()