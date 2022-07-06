import numpy as np
import pandas as pd
import glob
import json
import matplotlib.pyplot as plt

dir = "data/raw/0405/"
imu_folder = dir + 'imu/' #'data/raw/0405/imu/'
mqtt_folder = dir + 'mqtt/' #'data/raw/0405/mqtt/'

imu_file_list = sorted(glob.glob(imu_folder + "*.txt"))
mqtt_folder_list = sorted(glob.glob(mqtt_folder + "*/"))
imu_file = imu_file_list[0]

#get dataframe
f_imu = open(imu_file, "r")
df_current_file = pd.read_csv(f_imu, sep=" ", usecols=[0,2,3], names=["times","gyro_y", "gyro_z"])
df_current_file = df_current_file.dropna()

#filtering out standing parts
lines = open(mqtt_folder_list[0] + "dir/log000", "r").readlines()
com_start_time = json.loads(lines[0])["time"]/1000 #seconds
com_end_time = json.loads(lines[-1])["time"]/1000 #seconds
df_current_file = df_current_file.loc[(df_current_file['times'] > (com_start_time * (10 ** 3))+1000) & (df_current_file['times'] < ((com_end_time -  8.2) * (10 ** 3)))]

ft_y = np.fft.fft(np.array(df_current_file["gyro_y"]))
ft_z = np.fft.fft(np.array(df_current_file["gyro_z"]))

print(ft_y[0])

plt.plot(ft_y)
plt.show()