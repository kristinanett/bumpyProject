import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json

parser = argparse.ArgumentParser(description="Plot mqtt data from a file.")
parser.add_argument("file", help="Input file.") #for example "/data/raw/1506/mqtt/06-15-13-33/dir/log000"

args = parser.parse_args()

# JSON file
f = open(args.file, "r")
 
times = []
zs = []
xs = []

first = True
for line in f:
    if first == True:
        first_time = json.loads(line)["time"]/1000
        first = False
    times.append((json.loads(line)["time"]/1000)-first_time)
    zs.append(json.loads(line)["message"]["twist"]["angular"]["z"])
    xs.append(json.loads(line)["message"]["twist"]["linear"]["x"])

#plotting the data
fig, ax = plt.subplots(2, 1)

ax[0].plot(times[:3000], xs[:3000])
ax[0].set_ylim(0.45, 0.55)
#ax[0].set_xlabel('time (s)', fontsize = 12)
ax[0].set_ylabel('linear velocity (m/s)', fontsize = 12)
ax[0].grid(True)

ax[1].plot(times[:3000], zs[:3000])
ax[1].set_ylim(-0.5, 0.5)
ax[1].set_xlabel('time (s)', fontsize = 12)
ax[1].set_ylabel('wheel angle (rad)', fontsize = 12)
ax[1].grid(True)

fig.tight_layout()
plt.show()

# Closing file
f.close()