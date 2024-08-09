import os
import glob
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset

class GenericDataset(Dataset):
    """Generic dataset to train neural networks"""

    def __init__(self, root='./data/', extension='*.jpg', transform=None, train=True, limit_train=0.9):
        """
        Args:
            root (string): Directory with all the images.
            extension (string): Extension of the images to load.
            transform (callable, optional): Optional transform to be applied on a sample.
            train (bool): If True, creates dataset from training set, otherwise creates from test set.
            limit_train (float): Ratio of the dataset to use for training.
        """
        np.random.seed(0)
        self.transform = transform
        self.data = np.array(sorted(glob.glob(os.path.join(root, extension))))

        limit_train = int(len(self.data) * limit_train)
        indices = np.random.permutation(len(self.data))
        training_idx, test_idx = indices[:limit_train], indices[limit_train:]

        if train:
            self.data = self.data[training_idx]
        else:
            self.data = self.data[test_idx]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.data[idx]
        img = Image.open(filename)
        if self.transform:
            img = self.transform(img)
        return img, os.path.basename(filename)
