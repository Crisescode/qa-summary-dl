import tensorflow as tf

from .layers import Encoder, BahdanauAttention, Decoder
from ..utils.config import save_wv_model_path
from ..utils.wv_loader import load_embedding_matrix, get_vocab


class SeqSeq(tf.keras.Model):
    pass
