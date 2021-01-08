from torch.utils.data.sampler import SubsetRandomSampler
import torch
import numpy as np
from utils import dataset_wind


def get_train_valid_loader(data_dir,
                           input_timesteps,
                           prediction_timestep,
                           batch_size,
                           random_seed,
                           test_size=8813,
                           city_num=29,
                           city_idx=None,
                           feature_num=6,
                           feature_idx=None,
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=False):
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    # load the dataset
    train_dataset = dataset_wind.wind_dataset_us(
        filename=data_dir, inputTimesteps=input_timesteps, predictTimestep=prediction_timestep, train=True,
        test_size=test_size, city_idx=city_idx, feature_idx=feature_idx, feature_num=feature_num, city_num=city_num
    )

    valid_dataset = dataset_wind.wind_dataset_us(
        filename=data_dir, inputTimesteps=input_timesteps, predictTimestep=prediction_timestep, train=True,
        test_size=test_size, city_idx=city_idx, feature_idx=feature_idx, feature_num=feature_num, city_num=city_num
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return train_loader, valid_loader


def get_test_loader(data_dir,
                    input_timesteps,
                    prediction_timestep,
                    batch_size,
                    test_size=500,
                    shuffle=False,
                    city_num=29,
                    feature_num=11,
                    city_idx=None,
                    feature_idx=None,
                    num_workers=4,
                    pin_memory=False):
    dataset = dataset_wind.wind_dataset_us(
        filename=data_dir, inputTimesteps=input_timesteps, predictTimestep=prediction_timestep, train=False,
        test_size=test_size, city_idx=city_idx, feature_idx=feature_idx, feature_num=feature_num, city_num=city_num
    )

    data_loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=shuffle,
        num_workers=num_workers, pin_memory=pin_memory,
    )

    return data_loader
