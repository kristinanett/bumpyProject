import torch
from src.models.bumpy_dataset import BumpyDataset
from src.models.bumpy_dataset import Rescale, NormalizeIMG, ToTensor, Crop
from src.models.model import Net
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np
import hydra
from hydra.utils import get_original_cwd
import logging
import wandb
import omegaconf

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

     #wandb setup
     myconfig = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True) 
     wandb.init(config = myconfig, project='bumpyProject', group = "lr experiments", notes=train_params.comment)

     model = Net(model_params)
     if torch.cuda.is_available():
          log.info('##Converting network to cuda-enabled##')
          model.cuda()

     #initialize the train and validation set
     # dataset = BumpyDataset(
     #      get_original_cwd() + "/" + cfg.train.hyperparams.csv_data_path, 
     #      get_original_cwd() + "/" + train_params.img_data_path, 
     #      transform=transforms.Compose([Rescale(train_params.img_rescale), Crop(train_params.crop_ratio), NormalizeIMG(), ToTensor()])
     #      )

     #when running on dtu hpc (also change train config data paths)
     dataset = BumpyDataset(
          train_params.csv_data_path, 
          train_params.img_data_path, 
          transform=transforms.Compose([Rescale(train_params.img_rescale), Crop(train_params.crop_ratio), NormalizeIMG(), ToTensor()])
          )

     train_size = int(0.8 * len(dataset)) #10433 
     val_size = int(0.15*len(dataset)) #2609
     test_size = len(dataset) - train_size - val_size
     train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator = torch.manual_seed(cfg.train.hyperparams.seed))

     train_loader = DataLoader(train_set, batch_size=train_params.batch_size, shuffle=train_params.shuffle, num_workers=4, drop_last = True)
     val_loader = DataLoader(val_set, batch_size=train_params.batch_size, shuffle=train_params.shuffle, num_workers=4, drop_last = True)

     #check shapes
     x, y, idx = next(iter(train_loader))
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
               inputs, labels, idx = data
               # print(inputs[1])
               # print(labels)

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
                    wandb.log({"train loss": train_loss / 10})
                    train_losses.append(train_loss / 10)  # (loss.data.numpy())
                    train_loss = 0.0

          val_loss = 0.0
          val_loss_wandb = []
          model.eval()

          for i, data in enumerate(val_loader, 0):
               # data contains inputs, labels and inputs is a list of [images, commands]
               inputs, labels, idx = data

               if torch.cuda.is_available():
                    inputs, labels = [inputs[0].cuda(), inputs[1].cuda()] , labels.cuda()

               # forward pass
               output = model(inputs)

               # compute loss
               batch_loss = criterion(output, labels)

               # print statistics
               val_loss += batch_loss.item()  # loss.data[0] loss.detach().numpy()
               val_loss_wandb.append(batch_loss.item())

               if i % 10 == 9:  # print and save average validation loss for every epoch
                    log.info("[%d, %5d] validation loss: %.3f" % (epoch + 1, i + 1, val_loss / 10))
                    val_losses.append(val_loss / 10)  # (loss.data.numpy())
                    val_loss = 0.0

          mean_loss_epoch = np.mean(np.array(val_loss_wandb))
          wandb.log({"validation loss": mean_loss_epoch})
          val_loss_wandb = []
          
     # create directory if it does no exist already and save model
     os.makedirs("models/", exist_ok=True)
     torch.save(model.state_dict(), "models/thismodel.pt")

     log.info("Finished Training")
     plt.figure(figsize=(9, 9))
     nrofbatches_train = len(train_set)/train_params.batch_size
     nrofbatches_val = (len(val_set)/train_params.batch_size) #add ceil if using the last non-complete batch
     nrofsavings_val = np.floor(nrofbatches_val / 10.0)
     val_losses_per_epoch = np.mean(np.array(val_losses).reshape(-1, int(nrofsavings_val)), axis=1)

     x_train = np.arange(1, len(train_losses)+1)*10
     x_val = np.arange(1, train_params.num_epoch+1)*nrofbatches_train

     plt.plot(x_train, np.array(train_losses), 'r', marker='o', linestyle='-', label="Training Error")
     plt.plot(x_val, np.array(val_losses_per_epoch), 'b', marker='o', linestyle='-', label="Validation Error")
     plt.grid()
     plt.legend(fontsize=20)
     plt.xlabel("Train step", fontsize=20)
     plt.ylabel("Error", fontsize=20)
     ax = plt.gca()
     ax.set_ylim([0, 2.0])
     os.makedirs("reports/figures/", exist_ok=True)
     plt.savefig("reports/figures/training_curve1.png")

@hydra.main(config_path= "../conf", config_name="default_config.yaml")
def main(cfg):

     log.info(cfg.train.hyperparams.comment)
     torch.manual_seed(cfg.train.hyperparams.seed)
     train(cfg)

if __name__ == "__main__":
     main()