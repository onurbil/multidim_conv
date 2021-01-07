import torch
from torch import nn, optim
import torch.nn.functional as F
import time
from utils import data_loader_wind_us
from models import wind_models
from tqdm import tqdm
import scipy.io as sio
from torchsummary import summary


class FlatLayer(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


def count_parameters(model):
    print("model_summary")
    print("Layer_name" + "\t" * 7 + "Number of Parameters")
    print("=" * 100)
    model_parameters = [layer for layer in model.parameters() if layer.requires_grad]
    layer_name = [child for child in model.children()]
    j = 0
    total_params = 0
    print("\t" * 10)
    for i in layer_name:
        try:
            bias = (i.bias is not None)
        except:
            bias = False
        if not bias:
            param = model_parameters[j].numel() + model_parameters[j + 1].numel()
            j = j + 2
        else:
            param = model_parameters[j].numel()
            j = j + 1
        print(str(i) + "\t" * 3 + str(param))
        total_params += param
    print("=" * 100)
    print(f"Total Params:{total_params}")
    # return sum(p.numel() for p in model.parameters() if p.requires_grad)


def loss_batch(model, loss_func, xb, yb, opt=None):
    loss = loss_func(model(xb), yb)

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss.item(), len(xb)


def loss_batch_scaled(model, loss_func, xb, yb, opt=None, scaler=None):
    loss = loss_func(model(xb)*scaler, yb*scaler, reduction="none")

    if opt is not None:
        loss.backward()
        opt.step()
        opt.zero_grad()

    return loss, len(xb)


def test_model(data_folder, model, loss_func, dev="cpu", scaler=None):
    test_dl = data_loader_wind_us.get_test_loader(data_folder,
                                               batch_size=64,
                                               num_workers=4,
                                               pin_memory=True if dev == torch.device("cuda") else False)
    # Calc validation loss
    test_loss = torch.zeros(3, device=dev)
    test_num = 0
    model.eval()
    with torch.no_grad():
        for xb, yb in tqdm(test_dl, desc="Test"):
            if scaler is None:
                loss, num = loss_batch(model, loss_func, xb.to(dev), yb.to(dev))
            else:
                loss, num = loss_batch_scaled(model, loss_func, xb.to(dev), yb.to(dev), scaler=scaler)
            test_loss += torch.sum(loss, dim=0)
            test_num += num
        test_loss /= test_num

    print(f"Test loss: {test_loss}")


def train_wind_us(data_folder, epochs, input_timesteps, prediction_timestep, num_cities, num_features, city_idx=None,
                  feature_idx=None, dev=torch.device("cpu"), earlystopping=None):

    print(f"Device: {dev}")

    train_dl, valid_dl = data_loader_wind_us.get_train_valid_loader(data_folder,
                                                                    input_timesteps=input_timesteps,
                                                                    prediction_timestep=prediction_timestep,
                                                                    batch_size=64,
                                                                    random_seed=1337,
                                                                    city_idx=city_idx,
                                                                    feature_idx=feature_idx,
                                                                    valid_size=0.1,
                                                                    shuffle=True,
                                                                    num_workers=16,
                                                                    pin_memory=True if dev == torch.device("cuda") else False)
    if city_idx and feature_idx:
        num_output_channel = 1
    elif city_idx:
        num_output_channel = num_features
    elif feature_idx:
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


if __name__ == "__main__":
    dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    # torch.backends.cudnn.benchmark = True

    print("Weather dataset. Step: ", 4)
    data = "../processed_dataset/dataset_tensor.npy"
    train_wind_us(data, num_cities=29, num_features=11, city_idx=4, epochs=100, input_timesteps=6,
                  prediction_timestep=4, dev=dev, earlystopping=20)


    ### Test the newly trained model ###
    # load the model architecture and the weights
    # loaded = torch.load("models/wind_model.pt")

    # loaded = torch.load("models/checkpoints/best_val_loss_model_CNN3DWind.pt")
    # model = loaded["model"]
    # model.load_state_dict(loaded["state_dict"])
    # model.to(dev)
    # # get the scaler of the corresponding dataset
    # scaler = torch.as_tensor(sio.loadmat(f"{folder}Wind_data/lag=4/scale4.mat")["y_max_tr"], device=dev)
    # test_model(folder+data, model, F.l1_loss, dev, scaler=scaler)


