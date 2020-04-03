import os
import numpy as np
from torch.utils.data import Dataset


class SEMData(Dataset):
    def __init__(self, root_path, train=True):
        if train:
            print("Loading Training Dataset ...")
            path = os.path.join(root_path, "sem-task8", "train")
        else:
            print("Loading Testing Dataset ...")
            path = os.path.join(root_path, "sem-task8", "test")
        print(path)
        self.word_features = np.load(os.path.join(path, "word_feature.npy"))
        self.left_pf = np.load(os.path.join(path, "left_pf.npy"))
        self.right_pf = np.load(os.path.join(path, "right_pf.npy"))
        self.lexical_features = np.load(os.path.join(path, "lexical_feature.npy"))
        self.labels = np.load(os.path.join(path, "labels.npy"))

        self.input = list(zip(self.word_features, self.left_pf, self.right_pf, self.lexical_features, self.labels))
        print("Loading finish .")

    def __getitem__(self, idx):
        assert idx < len(self.input)
        return self.input[idx]

    def __len__(self):
        return len(self.input)





