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
        self.data = np.array(sorted(glob.glob(os.path.join(root, '**', extension), recursive=True)))


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


class WoodsDataset(torch.utils.data.Dataset):
    """Fibers dataset. to train neural net"""

    def __init__(self, transform=None,train=True, root='./data/woods/',limit_train=0.5):
        np.random.seed(0)
        self.transform = transform
        self.data = np.array(sorted(glob.glob('{}*.jpg'.format(root))))

        limit_train = int(len(self.data)*limit_train)
        indices = np.random.permutation(len(self.data))
        # print("path ",path," len ",len(self.data)," limit_train ",limit_train)
        training_idx, test_idx = indices[:limit_train], indices[limit_train:]

        self.image_list = []
        if train:
            self.data = self.data[training_idx]
        else:
            self.data = self.data[test_idx]
        # print('train '+str(train),' ',len(self.data))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        filename = self.data[idx]
        img = Image.open(filename)
        img_t = self.transform(img)
        return img_t,'wood'