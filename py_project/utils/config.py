import os
import pathlib

root = pathlib.Path(os.path.abspath(__file__))

stop_words_path = os.path.join(root, "common_data", "stopwords", "哈工大停用词表.txt")

train_x_path = os.path.join(root, "gen_data", "train_X")
train_y_path = os.path.join(root, "gen_data", "train_Y")
test_x_path = os.path.join(root, "gen_data", "test_X")
