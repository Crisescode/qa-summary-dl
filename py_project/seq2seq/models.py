import tensorflow as tf

from .layers import Encoder, BahdanauAttention, Decoder
from ..utils.config import save_wv_model_path
from ..utils.wv_loader import load_embedding_matrix, get_vocab


class SeqSeq(tf.keras.Model):
    def __init__(self, params):
        super(SeqSeq, self).__init__()
        self.embedding_matrix = load_embedding_matrix()
        self.params = params
        self.encoder = Encoder(
            params["vocab_size"],
            params["embedding_size"],
            self.embedding_matrix,
            params["enc_units"],
            params["batch_size"]
        )
        self.attention = BahdanauAttention(params["attention_units"])
        self.decoder = Decoder()
