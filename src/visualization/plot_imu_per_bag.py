import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser(description="Plot IMU data from a txt file.")
parser.add_argument("txt_file", help="Input txt file.")

args = parser.parse_args()

f = open(args.txt_file, "r")
df = pd.read_csv(f, sep=" ", usecols=[0,1,2,3,4,5,6], names=["times", "gyro_x", "gyro_y", "gyro_z", "acc_x", "acc_y", "acc_z"])

plt.subplot(2, 3, 1)
plt.plot(df["times"]/1000, df["gyro_x"])

plt.subplot(2, 3, 2)
plt.plot(df["times"]/1000, df["gyro_y"])
#plt.xlabel("Relative time (s)")
#plt.ylabel("Gyro y-axis data (deg/s)")
#plt.grid()

plt.subplot(2, 3, 3)
plt.plot(df["times"]/1000, df["gyro_z"])

plt.subplot(2, 3, 4)
plt.plot(df["times"]/1000, df["acc_x"])

plt.subplot(2, 3, 5)
plt.plot(df["times"]/1000, df["acc_y"])

plt.subplot(2, 3, 6)
plt.plot(df["times"]/1000, df["acc_z"])

plt.savefig('reports/figures/imu_plotx.png')
plt.show()



