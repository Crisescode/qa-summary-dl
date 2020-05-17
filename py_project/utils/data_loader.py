import numpy as np
from .config import train_x_path, train_y_path, test_x_path, stop_words_path


def load_dataset():
    """
    :return:
    """
    train_X = np.loadtxt(train_x_path)
    train_Y = np.loadtxt(train_y_path)
    test_X = np.loadtxt(test_x_path)

    train_X.dtype = "float64"
    train_Y.dtype = "float64"
    test_X.dtype = "float64"

    return train_X, train_Y, test_X


def load_stop_words(path):
    with open(path, 'r', encoding="utf-8") as f:
        stop_words = [word.strip() for word in f.readlines()]

    return stop_words


stop_words = load_stop_words(stop_words_path)


def clean_sentence(sentence):
    import re
    if isinstance(sentence, str):
        return re.sub(
            r'[\s+\-\|\!\/\[\]\{\}_,.$%^*(+\"\')]+|[:：+——()?【】“”！，。？、~@#￥%……&*（）]+|车主说|技师说|语音|图片|你好|您好',
            '', sentence)
    else:
        return ""


def filter_stopwords(words):
    return [word for word in words if word not in stop_words]


def sentence_proc(sentence):
    import jieba
    sentence = clean_sentence(sentence)
    words = jieba.cut(sentence)
    words = filter_stopwords(words)

    return " ".join(words)


def pad_proc(sentence, max_len, vocab):
    words = sentence.strip().split(" ")
    words = words[:max_len]

    sentence = [word if word in vocab else "<UNK>" for word in words]
    sentence = ["<START>"] + sentence + ["<STOP>"]
    sentence = sentence + ["<PAD>"] * (max_len - len(words))

    return " ".join(sentence)


def transform_data(sentence, vocab):
    words = sentence.split(" ")
    ids = [vocab[word] if word in vocab else vocab["<UNK>"] for word in words]

    return ids


def preprocess_sentence(sentence, max_len, vocab):
    # 切词处理
    sentence = sentence_proc(sentence)
    # 填充处理
    sentence = pad_proc(sentence, max_len, vocab)
    # 数值化处理
    sentence = transform_data(sentence, vocab)

    return np.array([sentence])
