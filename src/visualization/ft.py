import numpy as np
import pandas as pd
import glob
import json
import matplotlib.pyplot as plt

plt.rcParams['axes.grid'] = True

dir = "data/raw/0405/"
imu_folder = dir + 'imu/' #'data/raw/0405/imu/'
mqtt_folder = dir + 'mqtt/' #'data/raw/0405/mqtt/'

imu_file_list = sorted(glob.glob(imu_folder + "*.txt"))
mqtt_folder_list = sorted(glob.glob(mqtt_folder + "*/"))

def getDF(file_idx):
    #get dataframe
    f_imu = open(imu_file_list[file_idx], "r")
    df_current_file = pd.read_csv(f_imu, sep=" ", usecols=[0,2,3], names=["times","gyro_y", "gyro_z"])
    df_current_file = df_current_file.dropna()

    #filtering out standing parts
    lines = open(mqtt_folder_list[file_idx] + "dir/log000", "r").readlines()
    com_start_time = json.loads(lines[0])["time"]/1000 #seconds
    com_end_time = json.loads(lines[-1])["time"]/1000 #seconds
    df_current_file = df_current_file.loc[(df_current_file['times'] > (com_start_time * (10 ** 3))+1000) & (df_current_file['times'] < ((com_end_time -  8.2) * (10 ** 3)))]
    return df_current_file

df1 = getDF(0)
df2 = getDF(5)
print("The lengths are:", len(df1), len(df2)) #The lengths are:  10 924 and 10 699

ft1_y = np.fft.rfft(np.array(df1["gyro_y"]))
ft1_z = np.fft.rfft(np.array(df1["gyro_z"]))

ft2_y = np.fft.rfft(np.array(df2["gyro_y"]))
ft2_z = np.fft.rfft(np.array(df2["gyro_z"]))

freq1 = np.fft.rfftfreq(df1.shape[0])
freq2 = np.fft.rfftfreq(df2.shape[0])

# print(ft1_y[:5])
# print(ft2_y[:5])
freq = np.concatenate([freq1, freq2])
ft_y = np.concatenate([ft1_y, ft2_y]) 

#plotting
fig, ax = plt.subplots(1, 3)
ax[0].plot(freq1, ft1_y)
ax[0].set_ylabel('Amplitude', fontsize = 12)
ax[0].set_ylim(-1000, 1000)

ax[1].plot(freq1, ft1_z)
ax[1].set_xlabel('frequency (Hz)', fontsize = 12)
ax[1].set_ylim(-1000, 1000)

ax[2].plot(freq, ft_y)
ax[2].set_ylim(-1000, 1000)

plt.show()