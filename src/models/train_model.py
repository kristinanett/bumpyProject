import torch
from src.models.bumpy_dataset import BumpyDataset
from src.models.bumpy_dataset import Rescale, Normalize, ToTensor
from src.models.model import Net
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np
import sys
import hydra
from hydra.utils import get_original_cwd
import logging

#create dataset and dataloader
#dataset = BumpyDataset("data/processed/data.csv","data/processed", transform=transforms.Compose([Normalize(), ToTensor()]))
#dataloader = DataLoader(dataset, batch_size=1)
#dataloader_iter = iter(dataloader)

#A testrun for the raining code
#model = Net()
#model.train()
    
#for i in range(1):
#     x, y = next(dataloader_iter)

#outputs = model(x)

log = logging.getLogger(__name__)

def train(cfg):
     """Function for training the model in model.py and saving the resulting state dict in models.
     Also produces a training loss graph which is saved in reports/figures"""
     train_params = cfg.train.hyperparams
     model_params = cfg.model.hyperparams

     model = Net(model_params)
     if torch.cuda.is_available():
          log.info('##Converting network to cuda-enabled##')
          model.cuda()

     #initialize the train and validation set
     dataset = BumpyDataset(
          get_original_cwd() + "/" + cfg.train.hyperparams.csv_data_path, 
          get_original_cwd() + "/" + train_params.img_data_path, 
          transform=transforms.Compose([Rescale(train_params.img_rescale), Normalize(), ToTensor()])
          )
     
     train_size = int(0.8 * len(dataset)) #10433 
     val_size = len(dataset) - train_size #2609
     train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])

     train_loader = DataLoader(train_set, batch_size=train_params.batch_size, shuffle=train_params.shuffle, num_workers=2)
     #train_loader = DataLoader(dataset, batch_size=train_params.batch_size, shuffle = False) #for testing the training without shuffling
     val_loader = DataLoader(val_set, batch_size=train_params.batch_size, shuffle=True, num_workers=2)

     #check shapes
     x, y = next(iter(train_loader))
     log.info(f"Image batch dimension [B x C x H x W]: {x[0].shape}")
     log.info(f"Command batch dimension [B x L x Hin]: {x[1].shape}")

     num_epoch = train_params.num_epoch
     criterion = nn.MSELoss()
     optimizer = torch.optim.Adam(model.parameters(), lr=train_params.lr)
     train_losses, val_losses = [], []

     # training loop
     log.info("Starting training")
     for epoch in range(num_epoch):

          train_loss = 0.0
          model.train()

          for i, data in enumerate(train_loader, 0):
               # data contains inputs, labels and inputs is a list of [images, commands]
               inputs, labels = data

               if torch.cuda.is_available():
                    inputs, labels = [inputs[0].cuda(), inputs[1].cuda()] , labels.cuda()

               # zero the parameter gradients
               optimizer.zero_grad()

               # forward
               output = model(inputs)

               # compute gradients given loss
               batch_loss = criterion(output, labels)
               batch_loss.backward()
               optimizer.step()

               # print statistics
               train_loss += batch_loss.item()  # loss.data[0] loss.detach().numpy()

               if i % 10 == 9:  # print and save training loss every 10 batches
                    log.info("[%d, %5d] train loss: %.3f" % (epoch + 1, i + 1, train_loss / 10))
                    train_losses.append(train_loss / 10)  # (loss.data.numpy())
                    train_loss = 0.0

          # val_loss = 0.0
          # model.eval()

          # for i, data in enumerate(val_loader, 0):
          #      # data contains inputs, labels and inputs is a list of [images, commands]
          #      inputs, labels = data

          #      #if torch.cuda.is_available():
          #      #     inputs, labels = inputs.cuda(), labels.cuda()

          #      # forward pass
          #      output = model(inputs)

          #      # compute loss
          #      batch_loss = criterion(output, labels)

          #      # print statistics
          #      val_loss += batch_loss.item()  # loss.data[0] loss.detach().numpy()

          #      if i % 10 == 9:  # print and save validation loss every 10 batches
          #           log.info("[%d, %5d] validation loss: %.3f" % (epoch + 1, i + 1, val_loss / 10))
          #           val_losses.append(val_loss / 10)  # (loss.data.numpy())
          #           val_loss = 0.0
          

     # create directory if it does no exist already and save model
     os.makedirs("models/", exist_ok=True)
     torch.save(model.state_dict(), "models/thismodel.pt")

     log.info("Finished Training")
     plt.figure(figsize=(9, 9))
     plt.plot(np.array(train_losses), 'r', label="Training Error")
     #plt.plot(np.array(val_losses), 'b', label="Validation Error")
     plt.legend(fontsize=20)
     plt.xlabel("Train step", fontsize=20)
     plt.ylabel("Error", fontsize=20)
     os.makedirs("reports/figures/", exist_ok=True)
     plt.savefig("reports/figures/training_curve1.png")


@hydra.main(config_path= "../conf", config_name="default_config.yaml")
def main(cfg):
     torch.manual_seed(cfg.train.hyperparams.seed)
     train(cfg)


if __name__ == "__main__":
     main()