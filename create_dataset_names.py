import os

train_dir = "data/sample/train"
class_names = sorted(os.listdir(train_dir))

with open("class_names.txt", "w") as f:
    for name in class_names:
        f.write(name + "\n")