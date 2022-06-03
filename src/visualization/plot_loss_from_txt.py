import numpy as np 
import matplotlib.pyplot as plt
import os

#txt file
f = open("outputs/2022-06-03/19-01-30/train_model.log", "r")
 
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

train_losses = np.array(train_losses).flatten()
val_losses = np.array(val_losses).flatten()

print(np.shape(train_losses)) # (1160,)
print(np.shape(val_losses)) # (19,)

x_train = np.arange(0, np.shape(train_losses)[0]*10, 10)
x_val = np.arange(580, np.shape(train_losses)[0]*10, (np.shape(train_losses)[0]*10)/(np.shape(val_losses)[0]+1)) #int(11600/20))

plt.figure(figsize=(9, 9))
plt.plot(x_train, train_losses, 'r', marker='o', linestyle='-', label="Training Error")
plt.plot(x_val, val_losses, 'b', marker='o', linestyle='-', label="Validation Error")
plt.grid()
plt.legend(fontsize=20)
plt.xlabel("Train step", fontsize=20)
plt.ylabel("Error", fontsize=20)
ax = plt.gca()
ax.set_ylim([0, 1.4])
#plt.show()
os.makedirs("outputs/2022-06-03/19-01-30/reports/figures/", exist_ok=True)
plt.savefig("outputs/2022-06-03/19-01-30/reports/figures/training_curve1.png")
