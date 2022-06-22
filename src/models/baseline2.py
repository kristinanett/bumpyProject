import torch
from src.models.bumpy_dataset2 import BumpyDataset
from src.models.bumpy_dataset2 import Rescale, NormalizeIMG, ToTensor, Crop
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
import logging
import hydra
from hydra.utils import get_original_cwd
import wandb
import omegaconf

log = logging.getLogger(__name__)

def train_baseline(cfg):

    train_params = cfg.train.hyperparams

    #wandb setup
    myconfig = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True) 
    wandb.init(config = myconfig, project='bumpyProject', group = train_params.exp_group, notes=train_params.comment)

    dataset = BumpyDataset(
            train_params.csv_data_path, 
            train_params.img_data_path, 
            transform=transforms.Compose([Rescale(train_params.img_rescale), Crop(train_params.crop_ratio), NormalizeIMG(), ToTensor()])
            )

    train_size = int(0.8 * len(dataset)) #10433 
    val_size = len(dataset) - train_size #2609
    train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size], generator = torch.manual_seed(cfg.train.hyperparams.seed))
    val_loader = DataLoader(val_set, batch_size=train_params.batch_size, shuffle=True, num_workers=4, drop_last = True)

    df = pd.read_csv(train_params.csv_data_path, header=0)
    imu_all = np.array([df.iloc[:, 16:]])
    imu_mean, imu_std = np.mean(imu_all), np.std(imu_all)
    imu_standard = (imu_all-imu_mean)/imu_std
    imu_standard_mean = np.mean(imu_standard)

    num_epoch = train_params.num_epoch
    criterion = nn.MSELoss()
    val_losses = []

    log.info("Starting baseline validation")
    for epoch in range(num_epoch):
        val_loss = 0.0
        val_loss_wandb = []

        for i, data in enumerate(val_loader, 0):
            # data contains inputs, labels and inputs is a list of [images, commands]
            inputs, labels, idx = data
            labels = torch.mean(labels, dim=1) #from [32, 8, 1] to [32, 1] -predicting only mean imu value for each path

            if torch.cuda.is_available():
                inputs, labels = [inputs[0].cuda(), inputs[1].cuda(), inputs[2].cuda()] , labels.cuda()

            # forward pass
            #output = torch.full((labels.size()[0], 8, 1), imu_standard_mean) #predicting 8 values per path
            output = torch.full((labels.size()[0], 1), imu_standard_mean) #predicting 1 value per path
            if torch.cuda.is_available():
                output = output.cuda()

            # compute loss
            batch_loss = criterion(output, labels)

            # print statistics
            val_loss += batch_loss.item()  # loss.data[0] loss.detach().numpy()
            val_loss_wandb.append(batch_loss.item())

            if i % 10 == 9:  # print and save validation loss every 10 batches
                log.info("[%d, %5d] validation loss: %.3f" % (epoch + 1, i + 1, val_loss / 10))
                val_losses.append(val_loss / 10)  # (loss.data.numpy())
                val_loss = 0.0

        mean_loss_epoch = np.mean(np.array(val_loss_wandb))
        wandb.log({"validation loss": mean_loss_epoch})
        val_loss_wandb = []

    log.info("Finished Training")
    avg = np.sum(np.array(val_losses))/len(val_losses)
    log.info("Average loss is", avg)
    plt.figure(figsize=(9, 9))
    nrofbatches_train = len(train_set)/batch_size
    nrofbatches_val = len(val_set)/batch_size
    nrofsavings_val = np.floor(nrofbatches_val / 10.0)
    val_losses_per_epoch = np.mean(np.array(val_losses).reshape(-1, int(nrofsavings_val)), axis=1)
    x_val = np.arange(1, num_epoch+1)*nrofbatches_train

    plt.plot(x_val, np.array(val_losses_per_epoch), 'b', marker='o', linestyle='-', label="Validation Error")
    plt.grid()
    plt.legend(fontsize=20)
    plt.xlabel("Train step", fontsize=20)
    plt.ylabel("Error", fontsize=20)
    os.makedirs("reports/figures/", exist_ok=True)
    plt.savefig("reports/figures/baseline_validation_curve1.png")

@hydra.main(config_path= "../conf", config_name="default_config.yaml")
def main(cfg):

     log.info(cfg.train.hyperparams.comment)
     torch.manual_seed(cfg.train.hyperparams.seed)
     train_baseline(cfg)

if __name__ == "__main__":
     main()