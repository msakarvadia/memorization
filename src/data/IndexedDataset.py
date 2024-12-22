"""
Implements a torch dataset wrapper that indexes individual training examples
"""

from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader


class IndexedDataset(Dataset):
    def __init__(self, data):
        self.data = data

    # return example index alongside x and y
    def __getitem__(self, index):
        X = self.data[index]

        return X, index

    def __len__(self):
        return len(self.data)
