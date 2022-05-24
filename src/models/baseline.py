import torch
from src.models.bumpy_dataset2 import BumpyDataset2
from src.models.bumpy_dataset2 import Rescale, Normalize, ToTensor
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd

csv_file_path = 'data/processed/data3.csv'
num_epoch = 20
batch_size = 32

dataset = BumpyDataset2(csv_file_path, 'data/processed/imgs/', transform=transforms.Compose([Rescale(122), Normalize(), ToTensor()]))
train_size = int(0.8 * len(dataset)) #10433 
val_size = len(dataset) - train_size #2609
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=4)

df = pd.read_csv(csv_file_path, header=0)
imu_all = np.array([df.iloc[:, 16:]])
imu_mean, imu_std = np.mean(imu_all), np.std(imu_all)
imu_standard = (imu_all-imu_mean)/imu_std
imu_standard_mean = np.mean(imu_standard)

criterion = nn.MSELoss()
val_losses = []

print("Starting training")
for epoch in range(num_epoch):
    val_loss = 0.0

    for i, data in enumerate(val_loader, 0):
        # data contains inputs, labels and inputs is a list of [images, commands]
        inputs, labels = data
        #print(labels.size())

        # if torch.cuda.is_available():
        #     inputs, labels = [inputs[0].cuda(), inputs[1].cuda()] , labels.cuda()

        # forward pass
        output = torch.full((labels.size()[0], 8, 1), imu_standard_mean)
        #print(output.size())

        # compute loss
        batch_loss = criterion(output, labels)

        # print statistics
        val_loss += batch_loss.item()  # loss.data[0] loss.detach().numpy()

        if i % 10 == 9:  # print and save validation loss every 10 batches
            print("[%d, %5d] baseline train loss: %.3f" % (epoch + 1, i + 1, val_loss / 10))
            val_losses.append(val_loss / 10)  # (loss.data.numpy())
            val_loss = 0.0
          

print("Finished Training")
avg = np.sum(np.array(val_losses))/len(val_losses)
print("Average loss is", avg)
plt.figure(figsize=(9, 9))
nrofbatches_train = len(train_set)/batch_size
nrofbatches_val = len(val_set)/batch_size
nrofsavings_val = np.floor(nrofbatches_val / 10.0)
val_losses_per_epoch = np.mean(np.array(val_losses).reshape(-1, int(nrofsavings_val)), axis=1)
x_val = np.arange(1, num_epoch+1)*nrofbatches_train

plt.plot(x_val, np.array(val_losses_per_epoch), 'b', marker='o', linestyle='-', label="Validation Error")
plt.legend(fontsize=20)
plt.xlabel("Train step", fontsize=20)
plt.ylabel("Error", fontsize=20)
os.makedirs("reports/figures/", exist_ok=True)
plt.savefig("reports/figures/baseline_validation_curve1.png")
