import os
import pathlib

root = pathlib.Path(os.path.abspath(__file__))

train_x_path = os.path.join(root, "gen_data", "train_X")
train_y_path = os.path.join(root, "gen_data", "train_Y")
test_x_path = os.path.join(root, "gen_data", "test_X")
