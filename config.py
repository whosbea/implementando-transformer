import numpy as np

# Reprodutibilidade
SEED = 42
np.random.seed(SEED)

# Dimensões do modelo
D_MODEL = 128
D_K = 128
D_FF = 256

# Estrutura
N_ENCODER_LAYERS = 2
N_DECODER_LAYERS = 2
BATCH_SIZE = 1

# Comprimentos das sequências
SRC_SEQ_LEN = 6
TGT_SEQ_LEN = 5

# Vocabulário fictício
VOCAB = {
    "<PAD>": 0,
    "<START>": 1,
    "<EOS>": 2,
    "eu": 3,
    "gosto": 4,
    "de": 5,
    "pinguins": 6,
    "muito": 7,
    "fim": 8,
}

VOCAB_SIZE = len(VOCAB)
ID_TO_TOKEN = {idx: token for token, idx in VOCAB.items()}

# Tokens especiais
PAD_TOKEN_ID = VOCAB["<PAD>"]
START_TOKEN_ID = VOCAB["<START>"]
EOS_TOKEN_ID = VOCAB["<EOS>"]