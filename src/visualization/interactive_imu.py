import random
from itertools import count
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

plt.style.use('fivethirtyeight')

x_vals = []
y_vals = []

index = count()

f = open('data/raw/0405/imu/2022-05-04-15-12_LOG_IMU.txt', "r")
data = pd.read_csv(f, sep=" ", usecols=[0,2,3], names=["times","gyro_y", "gyro_z"])
time0 = data["times"][0]

def animate(i):
    x_vals.append((data['times'][i]-time0)/1000)
    y_vals.append(data['gyro_y'][i])

    plt.cla()
    plt.plot(x_vals, y_vals, label='Gyro y axis')

    plt.legend(loc='upper left')
    plt.tight_layout()


ani = FuncAnimation(plt.gcf(), animate, interval=25)

plt.tight_layout()
plt.show()

#62 seconds long video