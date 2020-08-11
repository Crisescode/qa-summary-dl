import tensorflow as tf

from models import Seq2Seq
from train_helper import train_model
from utils.config import save_wv_model_path, checkpoint_dir
from utils.gpu import gpu_config
from utils.params import get_params
from utils.wv_loader import get_vocab


def train(params):
    # config gpu
    gpu_config()

    # get vocab
    vocab, _ = get_vocab(save_wv_model_path)
    params["vocab_size"] = len(vocab)

    # build model
    print("Building the model ... ")
    model = Seq2Seq(params)

    # save chpk
    ckpt = tf.train.Checkpoint(Seq2Seq=model)
    ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_dir, max_to_keep=5)

    # train model
    train_model(model, vocab, params, ckpt_manager)


if __name__ == "__main__":
    # get params
    params = get_params()

    import sys
    sys.path.append("/home/lcz/lenlp/qa-summary-dl/")

    # train
    train(params)
