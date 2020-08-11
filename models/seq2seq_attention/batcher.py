import tensorflow as tf

from utils.data_loader import load_train_dataset, load_test_dataset


def train_batch_generator(batch_size, max_enc_len=200, max_dec_len=50, sample_sum=None):
    train_X, train_Y = load_train_dataset(max_enc_len, max_dec_len)
    if sample_sum:
        train_X = train_X[:sample_sum]
        train_Y = train_Y[:sample_sum]

    dataset = tf.data.Dataset.from_tensor_slices((train_X, train_Y)).shuffle(len(train_X))
    dataset = dataset.batch(batch_size, drop_remainder=True)

    steps_per_epoch = len(train_X) // batch_size

    return dataset, steps_per_epoch


def test_batch_generator(batch_size, max_enc_len=200):
    test_X = load_test_dataset(max_enc_len)
    dataset = tf.data.Dataset.from_tensor_slices(test_X)
    dataset = dataset.batch(batch_size)

    steps_per_epoch = len(test_X) // batch_size

    return dataset, steps_per_epoch

