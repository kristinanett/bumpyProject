import torch
from src.models.bumpy_dataset import BumpyDataset
from src.models.bumpy_dataset import Normalize, ToTensor
from src.models.model import Net
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np


# Params for the network should be defined in this code instead (or with hydra)
# seq_dim = 8
# input_dim = 28
# hidden_dim = 100
# layer_dim = 1
# output_dim = 10

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

def train():
     """Function for training the model in model.py and saving the resulting state dict in models.
     Also produces a training loss graph which is saved in reports/figures"""

     print("Starting the training")
     model = Net()
     trainset = BumpyDataset("data/processed/data.csv","data/processed", transform=transforms.Compose([Normalize(), ToTensor()]))
     trainloader = DataLoader(trainset, batch_size=32, shuffle=True)
     x, y = next(iter(trainloader))
     print(f"Image batch dimension [B x C x H x W]: {x[0].shape}")
     print(f"Command batch dimension [B x L x Hin]: {x[1].shape}")
     num_epoch = 10
     criterion = nn.MSELoss()
     optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
     losses = []

     # training loop
     for epoch in range(num_epoch):

          running_loss = 0.0
          model.train()

          for i, data in enumerate(trainloader, 0):
               # get the inputs (data is a list of [inputs, labels])
               inputs, labels = data

               # zero the parameter gradients
               optimizer.zero_grad()

               # forward
               output = model(inputs)

               # compute gradients given loss
               loss = criterion(output, labels)
               loss.backward()
               optimizer.step()

               # print statistics
               running_loss += loss.item()  # loss.data[0] loss.detach().numpy()

               if i % 10 == 9:  # print every 10 mini-batches
                    print("[%d, %5d] loss: %.3f" % (epoch + 1, i + 1, running_loss / 10))
                    losses.append(running_loss / 10)  # (loss.data.numpy())
                    running_loss = 0.0


     # create directory if it does no exist already and save model
     os.makedirs("models/", exist_ok=True)
     torch.save(model.state_dict(), "models/thismodel.pt")

     print("Finished Training")
     plt.figure(figsize=(9, 9))
     plt.plot(np.array(losses), label="Training Error")
     plt.legend(fontsize=20)
     plt.xlabel("Train step", fontsize=20)
     plt.ylabel("Error", fontsize=20)
     os.makedirs("reports/figures/", exist_ok=True)
     plt.savefig("reports/figures/training_curve1.png")

if __name__ == "__main__":
     train()