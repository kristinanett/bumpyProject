import numpy as np
import pandas as pd
import glob
import json
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter, freqz

plt.rcParams['axes.grid'] = True

dir = "data/raw/0405/"
imu_folder = dir + 'imu/' #'data/raw/0405/imu/'
mqtt_folder = dir + 'mqtt/' #'data/raw/0405/mqtt/'

imu_file_list = sorted(glob.glob(imu_folder + "*.txt"))
mqtt_folder_list = sorted(glob.glob(mqtt_folder + "*/"))

def butter_lowpass_filter(data, cutoff, fs, order=5):
    b, a = butter(order, cutoff, fs=fs, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

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

df = getDF(0)
print("The length is:", len(df)) #The lengths are:  10 924 and 10 699

#taking fourier transforms of the file data #####################################
ft_y = np.fft.rfft(np.array(df["gyro_y"]))
ft_z = np.fft.rfft(np.array(df["gyro_z"]))
freq = np.fft.rfftfreq(df.shape[0], 0.025)

#implementing a lowpass filter ##################################################
order = 6
fs = 40.0       # sample rate, Hz
cutoff = 11
y = butter_lowpass_filter(df["gyro_y"], cutoff, fs, order)

#taking a fourier transform of the lowpassed data ###############################
ft_y_after = np.fft.rfft(np.array(y))
freq_after = np.fft.rfftfreq(len(y), 0.025)

#plotting #######################################################################
fig, ax = plt.subplots(1, 4)

#time domain before lowpass
ax[0].plot(df["gyro_y"])
ax[0].set_ylabel('IMU value', fontsize = 12)
ax[0].set_xlabel('Time', fontsize = 12)

#time domain after lowpass
ax[1].plot(y)
ax[1].set_ylabel('IMU value', fontsize = 12)
ax[1].set_xlabel('Time', fontsize = 12)

#frequency domain before lowpass
ax[2].plot(freq, ft_y.real)
ax[2].set_ylabel('Amplitude', fontsize = 12)
ax[2].set_xlabel('Frequency', fontsize = 12)
#ax[2].set_ylim(-1000, 1000)

#frequency domain after lowpass
ax[3].plot(freq_after, ft_y_after.real)
ax[3].set_ylabel('Amplitude', fontsize = 12)
ax[3].set_xlabel('Frequency', fontsize = 12)
#ax[3].set_ylim(-1000, 1000)

plt.show()