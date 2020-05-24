import os
import pathlib

root = pathlib.Path(os.path.abspath(__file__)).parent.parent.parent
print("root path:", root)

stop_words_path = os.path.join(root, "data", "stopwords", "哈工大停用词表.txt")

train_x_path = os.path.join(root, "data", "train_X")
train_y_path = os.path.join(root, "data", "train_Y")
test_x_path = os.path.join(root, "data", "test_X")

save_wv_model_path = os.path.join(root, "data", "word2vec.model")

embedding_matrix_path = os.path.join(root, "data", "embedding_matrix")

# 模型保存文件夹
checkpoint_dir = os.path.join(root, "data", 'checkpoints', 'training_checkpoints_mask_loss_dim300_seq')

checkpoint_prefix = os.path.join(checkpoint_dir, 'ckpt')
