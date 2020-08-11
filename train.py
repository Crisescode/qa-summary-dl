# -*- coding:utf-8 -*-

import tensorflow as tf
from seq2seq.models import Seq2Seq
from utils.google_config import config_gpu
from utils.wv_loader import get_vocab
from seq2seq.train_helper import train_model
from utils.google_config import save_wv_model_path, checkpoint_dir
from utils.params import get_params


def train(params):
    # gpu config
    config_gpu()

    # vocab
    vocab, _ = get_vocab(save_wv_model_path)
    params["vocab_size"] = len(vocab)

    # load model
    print("Building the model ... ")
    model = Seq2Seq(params)

    # save checkpoint
    checkpoint = tf.train.Checkpoint(Seq2Seq=model)
    checkpoint_mgr = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_step=5)

    # train model
    train_model(model, vocab, params, checkpoint_mgr)


if __name__ == "__main__":
    # params
    params = get_params()

    # train
    train(params)
