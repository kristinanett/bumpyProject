import numpy as np 
import matplotlib.pyplot as plt
import os
import pandas as pd
import csv
import wandb

#This file breaks with some files (yet to be fixed)
file_path =  "outputs/2022-06-03/19-01-30/train_model.log" #"outputs/2022-05-18/22-05-01/train_model.log"
f = open(file_path, "r")
 
train_losses = []
val_losses = []
val_losses_small = []

for line in f:
    if "train loss" in line:
        train_losses.append(float(line.split(" ")[-1].strip()))
    elif "validation loss" in line:
        if len(val_losses_small) == 10:
            val_losses.append(np.mean(val_losses_small))
            val_losses_small = []
            val_losses_small.append(float(line.split(" ")[-1].strip()))
        else:
            val_losses_small.append(float(line.split(" ")[-1].strip()))

train_losses = np.array(train_losses).flatten().tolist()
val_losses = np.array(val_losses).flatten().tolist()

print(np.shape(train_losses)) # (1160,) 620
print(np.shape(val_losses)) # (19,) 13

x_train = np.arange(0, np.shape(train_losses)[0]*10, 10).tolist()
x_val = np.arange(580, np.shape(train_losses)[0]*10, int((np.shape(train_losses)[0]*10)/(np.shape(val_losses)[0]+1))).tolist() #int(11600/20))
print(len(x_train)) #1160 620
print(len(x_val)) #19 14

# #trying to make a plot directly with wandb
# run = wandb.init(project="mnist-test-project", job_type="visualizations")
# data = [[x, y, z] for (x, y, z) in zip(range(len(train_losses)), train_losses, val_losses)]

# #table = wandb.Table(data=data, columns = ["step", "train_loss", "val_loss"])
# #line_plot = wandb.plot.line(table, x='step', y='train_loss', title='Line Plot')
# #histogram = wandb.plot.histogram(table, value='train_loss', title='Histogram')
# multiline_plot = wandb.plot.line_series(
#                        xs=[x_train, x_val],
#                        ys=[train_losses, val_losses],
#                        keys=["Train loss", "Val loss"], #go into the legend for normal line and dotted
#                        xname="Steps")

# #log only one plot
# wandb.log({"Comparison plot" : multiline_plot})
# # Log multiple plots
# # wandb.log({'line_1': line_plot, 
# #             'histogram_1': histogram})

# run.finish()

#plotting with matplotlib
# plt.figure(figsize=(9, 9))
# plt.plot(x_train, train_losses, 'r', marker='o', linestyle='-', label="Training Error")
# plt.plot(x_val, val_losses, 'b', marker='o', linestyle='-', label="Validation Error")
# plt.grid()
# plt.legend(fontsize=20)
# plt.xlabel("Train step", fontsize=20)
# plt.ylabel("Error", fontsize=20)
# ax = plt.gca()
# ax.set_ylim([0, 1.4])
# plt.show()
#os.makedirs("outputs/2022-06-03/19-01-30/reports/figures/", exist_ok=True)
#plt.savefig("outputs/2022-06-03/19-01-30/reports/figures/training_curve1.png")
