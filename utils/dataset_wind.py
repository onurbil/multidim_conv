import torch
from torch.utils.data import Dataset
import scipy.io as sio
import pickle
import numpy as np

class wind_dataset(Dataset):
    def __init__(self, filename, train: bool):
        if filename[-4:] == ".mat":
            mat = sio.loadmat(filename)
        else:
            mat = sio.loadmat(filename+".mat")

        if train:
            x = mat["Xtr"]
            y = mat["Ytr"]
        else:
            x = mat["Xtest"]
            y = mat["Ytest"]

        self.x = torch.as_tensor(x).float()
        self.y = torch.as_tensor(y).float()

    def __getitem__(self, item):
        return self.x[item], self.y[item]

    def __len__(self):
        return len(self.x)

class wind_dataset_us(Dataset):
    def __init__(self, filename, inputTimesteps, predictTimestep, train: bool, feature_idx=None, city_idx=None):


        data = np.load(filename, allow_pickle=True).astype(float)
        data = data[:, :29, :11]

        self.inputTimesteps = inputTimesteps
        self.predictTimestep = predictTimestep
        self.feature_idx = feature_idx
        self.city_idx = city_idx

        if train:
            x = data[:-8813]
        else:
            x = data[-8813:]
        self.x = torch.as_tensor(x).float()

    def __getitem__(self, item):
        # Target is the wind speed of all (7) cities. Wind speed is the first entry of the last dimension
        x = self.x[item:item + self.inputTimesteps]
        if self.city_idx and self.feature_idx:
            y = self.x[item + self.inputTimesteps + self.predictTimestep, self.city_idx, self.feature_idx]
        elif self.city_idx:
            y = self.x[item + self.inputTimesteps + self.predictTimestep, self.city_idx, :]
        elif self.feature_idx:
            y = self.x[item + self.inputTimesteps + self.predictTimestep, :, self.feature_idx]
        else:
            y = self.x[item + self.inputTimesteps + self.predictTimestep]
        y = torch.flatten(y)
        return x, y

    def __len__(self):
        # Number of total time steps - input - prediction
        return self.x.shape[0]-self.inputTimesteps-self.predictTimestep


class wind_dataset_nl(Dataset):
    def __init__(self, filename, inputTimesteps, predictTimestep, CTF, train: bool):
        if filename[-4:] == ".pkl":
            pass
        else:
            filename = filename+".pkl"
        with open(filename, "rb") as f:
            data = pickle.load(f)
        self.inputTimesteps = inputTimesteps
        self.predictTimestep = predictTimestep
        if train:
            x = data["train"]
        else:
            x = data["test"]
        # CTF = Cities x Timesteps x Features
        # If false, defaults to: Timesteps x Cities x Features
        self.CTF = CTF
        if self.CTF:
            x = np.transpose(x, (1, 0, 2))
        self.x = torch.as_tensor(x).float()

    def __getitem__(self, item):
        # Target is the wind speed of all (7) cities. Wind speed is the first entry of the last dimension
        if self.CTF:
            x = self.x[:, item:item + self.inputTimesteps]
            y = self.x[:, item + self.inputTimesteps + self.predictTimestep, 0]
        else:
            x = self.x[item:item + self.inputTimesteps]
            y = self.x[item + self.inputTimesteps + self.predictTimestep, :, 0]
        return x, y

    def __len__(self):
        # Number of total time steps - input - prediction
        if self.CTF:
            return self.x.shape[1]-self.inputTimesteps-self.predictTimestep
        else:
            return self.x.shape[0]-self.inputTimesteps-self.predictTimestep
