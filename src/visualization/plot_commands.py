import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse
import json

parser = argparse.ArgumentParser(description="Plot mqtt data from a file.")
parser.add_argument("file", help="Input file.")

args = parser.parse_args()

# JSON file
f = open(args.file, "r")
 
times = []
zs = []

for line in f:
    times.append(json.loads(line)["time"]/1000)
    zs.append(json.loads(line)["message"]["twist"]["angular"]["z"])

#plotting the data
fig, ax = plt.subplots()
ax.plot(zs, times)
ax.set_xlim(0.5, -0.5)
ax.set_xlabel('decreasing radians')
ax.set_ylabel('time')
ax.grid(True)

plt.show()

# Closing file
f.close()