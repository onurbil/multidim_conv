import torch
from torch import nn, optim
import torch.nn.functional as F
import time
from utils import data_loader_wind_us
from models import wind_models
from tqdm import tqdm
import scipy.io as sio
from torchsummary import summary
import matplotlib.pyplot as plt
import numpy as np



def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def plot_figure(pred,y):

    plt.figure(figsize=(14, 8))
    size = pred.shape[0]
    plt.plot(range(size), pred, label='pred')
    plt.plot(range(len(y)), y, label='true')
    plt.legend()
    plt.show()   


def train_wind_us(data_folder, epochs, input_timesteps, prediction_timestep, test_size, num_cities, num_features, city_idx=None,
                  feature_idx=None, dev=torch.device("cpu"), earlystopping=None):

    print(f"Device: {dev}")

    train_dl, valid_dl = data_loader_wind_us.get_train_valid_loader(data_folder,
                                                                    input_timesteps=input_timesteps,
                                                                    prediction_timestep=prediction_timestep,
                                                                    batch_size=64,
                                                                    random_seed=1337,
                                                                    test_size=test_size,
                                                                    city_num=num_cities,
                                                                    city_idx=city_idx,
                                                                    feature_num=num_features,
                                                                    feature_idx=feature_idx,
                                                                    valid_size=0.1,
                                                                    shuffle=True,
                                                                    num_workers=16,
                                                                    pin_memory=True if dev == torch.device("cuda") else False)
    if city_idx is not None and feature_idx is not None:
        num_output_channel = 1
    elif city_idx is not None:
        num_output_channel = num_features
    elif feature_idx is not None:
        num_output_channel = num_cities
    else:
        num_output_channel = num_cities * num_features

    ### Model definition ###
    model = wind_models.MultidimConvNetwork(channels=input_timesteps, height=num_features, width=num_cities,
                                            output_channels=num_output_channel, kernels_per_layer=16, hidden_neurons=128)

    # print("Parameters: ", sum(p.numel() for p in model.parameters() if p.requires_grad))
    summary(model, (input_timesteps, num_cities, num_features), device="cpu")
    # Put the model on GPU
    model.to(dev)
    # Define optimizer
    lr = 0.001
    opt = optim.Adam(model.parameters(), lr=lr)
    # Loss function
    # loss_func = F.mse_loss
    loss_func = F.l1_loss
    #### Training ####
    best_val_loss = 1e300

    earlystopping_counter = 0
    pbar = tqdm(range(epochs), desc="Epochs")
    for epoch in pbar:
        model.train()
        train_loss = 0.0
        total_num = 0

        for i, (xb, yb) in enumerate(train_dl):

            loss, num = loss_batch(model, loss_func, xb.to(dev), yb.to(dev), opt)
            if loss_func == F.l1_loss:
                num = 1
            train_loss += loss
            total_num += num
        train_loss /= total_num

        # Calc validation loss
        val_loss = 0.0
        val_num = 0
        model.eval()
        with torch.no_grad():
            for xb, yb in valid_dl:
                loss, num = loss_batch(model, loss_func, xb.to(dev), yb.to(dev))
                if loss_func == F.l1_loss:
                    num = 1
                val_loss += loss
                val_num += num
            val_loss /= val_num
    
        pbar.set_postfix({'train_loss': train_loss, 'val_loss': val_loss})
        
        # Save the model with the best validation loss
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            earlystopping_counter = 0
        
        else:
            if earlystopping is not None:
                earlystopping_counter += 1
                if earlystopping_counter >= earlystopping:
                    print(f"Stopping early --> val_loss has not decreased over {earlystopping} epochs")
                    break


    torch.save({ 'model': model, 'epoch': epoch, 
                 'state_dict': model.state_dict(), 
                 'optimizer_state_dict': opt.state_dict(), 
                 'val_loss': val_loss, 'train_loss': train_loss }, 
                  f"models/checkpoints/model_{model.__class__.__name__}.pt") 
    

 


if __name__ == "__main__":
    
    
    # Parameters:
    num_features = 11
    num_cities = 29
    city_idx = 0
    feature_idx = 4
    epochs = 200
    test_size = 2500
    train_model = False

    input_timesteps = 6
    prediction_timesteps = 4
    early_stopping = 20
    data = "../processed_dataset/dataset_tensor.npy"
    load_model_path = "models/checkpoints/model_MultidimConvNetwork.pt"
    
    
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # torch.backends.cudnn.benchmark = True

    print("Weather dataset. Step: ", 4)
    
    if train_model:
        train_wind_us(data, num_cities=num_cities, test_size=test_size, num_features=num_features, 
                            city_idx=city_idx, feature_idx=feature_idx, epochs=epochs, 
                            input_timesteps=input_timesteps, prediction_timestep=prediction_timesteps, 
                            dev=dev, earlystopping=early_stopping)


    ### Test the newly trained model ###
    # load the model architecture and the weights
    loaded = torch.load(load_model_path)
    model = loaded["model"]
    model.load_state_dict(loaded["state_dict"])
    model.to(dev)
    
    test_dl = data_loader_wind_us.get_test_loader(data,
                                                  input_timesteps=input_timesteps,
                                                  prediction_timestep=prediction_timesteps,
                                                  batch_size=test_size,
                                                  test_size=test_size,
                                                  feature_num=num_features,
                                                  city_num=num_cities,
                                                  city_idx=city_idx,
                                                  feature_idx=feature_idx,
                                                  shuffle=False,
                                                  num_workers=16,
                                                  pin_memory=True if dev == torch.device("cuda") else False)


    for x,y in test_dl:
        
        test_pred = model(x)
        test_loss = F.l1_loss(test_pred, y)
        print(test_loss.detach().numpy())
        plot_figure(test_pred.detach().numpy(), y.detach().numpy())
        
