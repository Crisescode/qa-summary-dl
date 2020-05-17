import os
import pathlib

root = pathlib.Path(os.path.abspath(__file__)).parent.parent.parent
print("root path:", root)

stop_words_path = os.path.join(root, "Hashing", "common_data", "stopwords", "哈工大停用词表.txt")

train_x_path = os.path.join(root, "Hashing", "gen_data", "train_X")
train_y_path = os.path.join(root, "Hashing", "gen_data", "train_Y")
test_x_path = os.path.join(root, "Hashing", "gen_data", "test_X")

save_wv_model_path = os.path.join(root, "gen_data", "word2vec.model")

embedding_matrix_path = os.path.join(root, "gen_data", "embedding_matrix")

