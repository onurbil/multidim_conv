import torch
from torch.utils.data import Dataset
import scipy.io as sio
import pickle
import numpy as np


class wind_dataset_us(Dataset):
    def __init__(self, filename, inputTimesteps, predictTimestep, train: bool, test_size=8813, feature_num=11,
                 feature_idx=None, city_num=29, city_idx=None):


        data = np.load(filename, allow_pickle=True).astype(float)
        data = data[:, :city_num, :feature_num]

        self.inputTimesteps = inputTimesteps
        self.predictTimestep = predictTimestep
        self.feature_idx = feature_idx
        self.city_idx = city_idx
        ignore = 800
        if train:
            x = data[:-test_size-ignore]
            
        else:
            x = data[-test_size-ignore:-ignore]

        self.x = torch.as_tensor(x).float()


    def __getitem__(self, item):

        x = self.x[item:item + self.inputTimesteps]
        if self.city_idx is not None and self.feature_idx is not None:
            y = self.x[item + self.inputTimesteps + self.predictTimestep, self.city_idx, self.feature_idx]
        elif self.city_idx is not None:
            y = self.x[item + self.inputTimesteps + self.predictTimestep, self.city_idx, :]
        elif self.feature_idx is not None:
            y = self.x[item + self.inputTimesteps + self.predictTimestep, :, self.feature_idx]
        else:
            y = self.x[item + self.inputTimesteps + self.predictTimestep]
        y = torch.flatten(y)
        return x, y

    def __len__(self):
        # Number of total time steps - input - prediction
        return self.x.shape[0]-self.inputTimesteps-self.predictTimestep
