# -*- coding:utf-8 -*-

import tensorflow as tf
from seq2seq.batcher import train_batch_generator


def train_model(model, vocab, params, ckpt_mgr):
    epochs = params["epochs"]
    batch_size = params["batch_size"]

    pad_index = vocab["<PAD>"]
    nuk_index = vocab["<UNK>"]
    start_index = vocab["<START>"]

    # vocab size
    params["vocab_size"] = len(vocab)

    optimizer = tf.keras.optimizers.Adam(name='Adam', learning_rate=0.001)
    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

    # loss function
    def loss_function(real, pred):
        pad_mask = tf.math.equal(real, pad_index)
        nuk_mask = tf.math.equal(real, nuk_index)
        mask = tf.math.logical_not(tf.math.logical_or(pad_mask, nuk_mask))
        loss_ = loss_object(real, pred)
        mask = tf.cast(mask, dtype=loss_.dtype)
        loss_ *= mask

        return tf.reduce_mean(loss_)

    # train
    @tf.function
    def train_step(enc_inp, dec_tag):
        batch_loss = 0
        with tf.GradientTape() as tape:
            # encoder
            enc_output, enc_hidden = model.call_encoder(enc_inp)

            dec_input = tf.expand_dims([start_index] * batch_size, 1)

            # decoder hidden input
            dec_hidden = enc_hidden

            prediction, _ = model(dec_input, dec_hidden, enc_output, dec_tag)

            batch_loss = loss_function(dec_tag[:, 1:], prediction)

            variables = model.encoder.trainable_variables + model.decoder.trainable_variables + model.attention.trainable_variables

            gradients = tape.gradient(batch_loss, variables)

            optimizer.apply_gradients(zip(gradients, variables))

            return batch_loss

    dataset, steps_per_epoch = train_batch_generator(batch_size)

    for epoch in range(epochs):
        import time
        start = time.time()
        total_loss = 0

        for (batch, (inputs, target)) in enumerate(dataset.take(steps_per_epoch)):
            batch_loss = train_step(inputs, target)
            total_loss += batch_loss

            if batch % 1 == 0:
                print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                             batch,
                                                             batch_loss.numpy()))
        # saving (checkpoint) the model every 2 epochs
        if (epoch + 1) % 2 == 0:
            ckpt_save_path = ckpt_mgr.save()
            print('Saving checkpoint for epoch {} at {}'.format(epoch + 1,
                                                                ckpt_save_path))

        print('Epoch {} Loss {:.4f}'.format(epoch + 1,
                                            total_loss / steps_per_epoch))
        print('Time taken for 1 epoch {} sec\n'.format(time.time() - start))