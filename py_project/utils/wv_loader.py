import logging
import numpy as np

from gensim.models.word2vec import LineSentence, Word2Vec

# from utils.config import embedding_matrix_path


logging.basicConfig(format="%(asctime)s : %(levelname)s : %(message)s", level=logging.INFO)


def get_embedding_matrix(save_wv_model_path):
    wv_model = Word2Vec.load(save_wv_model_path)
    embedding_matrix = wv_model.wv.vectors

    return embedding_matrix


def get_vocab(save_wv_model_path):
    wv_model = Word2Vec.load(save_wv_model_path)
    vocab = {word: index for index, word in enumerate(wv_model.wv.index2word)}
    reverse_vocab = {index:word for index, word in enumerate(wv_model.wv.index2word)}

    return vocab, reverse_vocab


def load_vocab(file_path):
    vocab = {}
    reverse_vocab = {}
    for line in open(file_path, "r", encoding='utf-8').readlines():
        word, index = line.strip().split("\t")
        index = int(index)
        vocab[word] = index
        reverse_vocab[index] = word
    return vocab, reverse_vocab


def load_embedding_matrix(embedding_matrix_path="/content/gdrive/My Drive/Colab Notebooks/Hashing/gen_data/embedding_matrix"):
    return np.load(embedding_matrix_path + ".npy")




