import torch
from src.models.bumpy_dataset import BumpyDataset
from src.models.bumpy_dataset import Rescale, Normalize, ToTensor
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torch.nn as nn
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from src.models.model import Net
from torch.profiler import profile, record_function, ProfilerActivity, tensorboard_trace_handler


csv_file_path = 'data/processed/data2.csv'
num_epoch = 20
batch_size = 32

dataset = BumpyDataset(csv_file_path, 'data/processed', transform=transforms.Compose([Rescale(122), Normalize(), ToTensor()]))
train_size = int(0.8 * len(dataset)) #10433 
val_size = len(dataset) - train_size #2609
train_set, val_set = torch.utils.data.random_split(dataset, [train_size, val_size])
val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True, num_workers=2)
val_loader_iter = iter(val_loader)

df = pd.read_csv(csv_file_path, header=0)
imu_all = np.array([df.iloc[:, 16:]])
imu_mean, imu_std = np.mean(imu_all), np.std(imu_all)
imu_standard = (imu_all-imu_mean)/imu_std
imu_standard_mean = np.mean(imu_standard)

criterion = nn.MSELoss()
val_losses = []
model = Net(model_params)

with profile(activities=[ProfilerActivity.CPU], record_shapes=True, on_trace_ready=tensorboard_trace_handler("reports/figures/profiling")) as prof:
    with record_function("model_inference"):
   # code that I want to profile
   
        
        for i in range(5):

            inputs, labels = next(val_loader_iter)
            output = torch.full((labels.size()[0], 8, 1), imu_standard_mean)
            batch_loss = criterion(output, labels)
            val_losses.append(batch_loss)


print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))


