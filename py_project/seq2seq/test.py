# -*- coding:utf-8 -*-
# copy right of Crise

import tensorflow as tf
from seq2seq.batcher import batcher
from seq2seq.models import Seq2Seq

from tqdm import tqdm
from seq2seq.test_helper import beam_decode
from utils.wv_loader import Vocab
from utils.params import get_params
from utils.config import checkpoint_dir


def test(params):
    assert params["mode"].lower() == "test", "change training mode to 'test' or 'eval'."
    assert params["beam_size"] == params["batch_size"], "Beam size must be equal to batch size, change the params."

    print("Building model ... ")
    model = Seq2Seq(params)

    print("Creating vocab ... ")
    vocab = Vocab(params["vocab_path"], params["vocab_size"])

    print("Creating batcher ... ")
    b = batcher(vocab, params)

    print("creating the checkpoint manager")
    checkpoint = tf.train.Checkpoint(Seq2Seq=model)
    checkpoint_manager = tf.train.CheckpointManager(checkpoint, checkpoint_dir, max_to_keep=5)
    checkpoint.restore(checkpoint_manager.latest_checkpoint)
    if checkpoint_manager.latest_checkpoint:
        print("Restored from {}".format(checkpoint_manager.latest_checkpoint))
    else:
        print("Initializing from scratch.")
    print("Model restored")

    for batch in b:
        print(beam_decode(model, batch, vocab, params))


def test_and_save(params):
    assert params["test_save_dir"], "provide a dir where to save the results."
    gen = test(params)
    with tqdm(total=params["num_to_test"], position=0, leave=True) as pbar:
        for i in range(params["num_to_test"]):
            trial = next(gen)
            with open(params["test_save_dir"] + "/article_" + str(i) + ".txt", "w") as f:
                f.write("article:\n")
                f.write(trial.text)
                f.write("\n\nabstract:\n")
                f.write(trial.abstract)
            pbar.update(1)


if __name__ == "__main__":
    # get params
    params = get_params()

    params["batch_size"] = 32
    params["mode"] = "test"
    test(params)
