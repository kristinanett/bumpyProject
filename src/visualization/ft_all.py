import glob
import json
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt

plt.rcParams['axes.grid'] = True

#this code only works on a folder of raw data
#this code assumes that every bag file has a corresponding mqtt file and there is exactly one mqtt log in each timestamped folder

class FT():
    def __init__(self, dir):
        
        """
        Objective:
            1. filter out images from beginning and end of bagfile based on the com file start and end.
            2. Take a Fourier transform of all the imu values by file then add them and plot them

        Args:
            dir (string): Directory with all the raw data to process eg:data/raw/0405/
        """

        self.image_topic = "/cam1/image_rect/compressed"
        
        self.cam_folder = dir + 'cam/' #'data/raw/0405/cam/'
        self.imu_folder = dir + 'imu/' #'data/raw/0405/imu/'
        self.mqtt_folder = dir + 'mqtt/' #'data/raw/0405/mqtt/'

        self.cam_file_list = sorted(glob.glob(self.cam_folder + "*.bag"))
        self.imu_file_list = sorted(glob.glob(self.imu_folder + "*.txt"))
        self.mqtt_folder_list = sorted(glob.glob(self.mqtt_folder + "*/"))

        #check if can access the raw data
        if not self.cam_file_list or not self.imu_file_list or not self.mqtt_folder_list:
            print("Unable to locate the raw data. Check the inserted path and try again")
            sys.exit(0)

    def transformBags(self):

        #looping over all the bag files in one day
        file_count=0
        ft_all_y, ft_all_z, freq_all = np.array([]), np.array([]), np.array([])
        for input_cam_file, com_folder, imu_file in zip(self.cam_file_list, self.mqtt_folder_list, self.imu_file_list):

            f_com = open(com_folder + "dir/log000", "r")
            f_imu = open(imu_file, "r")
            df = pd.read_csv(f_imu, sep=" ", usecols=[0,2,3], names=["times","gyro_y", "gyro_z"])
            df = df.dropna() #drop rows with nan values

            #accessing the velocity command document to get the time that commands started/stopped sending and saving (to filter out all images taken before/after)
            file_idx = self.cam_file_list.index(input_cam_file)
            mqtt_folder = self.mqtt_folder_list[file_idx]
            print(file_count+1, "/", len(self.cam_file_list), ":", "Bag file", input_cam_file, "\tMQTT folder", mqtt_folder)

            lines = open(mqtt_folder + "dir/log000", "r").readlines()
            com_start_time = json.loads(lines[0])["time"]/1000 #seconds
            com_end_time = json.loads(lines[-1])["time"]/1000 #seconds

            #filtering out the beginning and end of file from imu data
            df_filtered = df.loc[(df['times'] > (com_start_time * (10 ** 3))+1000) & (df['times'] > (df['times'][0]+1000)) & (df['times'] < ((com_end_time -  8.2) * (10 ** 3)))]

            #take Fourier transform of filtered imu data
            ft_y = np.fft.rfft(np.array(df_filtered["gyro_y"]))
            ft_z = np.fft.rfft(np.array(df_filtered["gyro_z"]))
            freq = np.fft.rfftfreq(df_filtered.shape[0])

            #add this file FT to the combined
            ft_all_y = np.concatenate([ft_all_y, ft_y])
            ft_all_z = np.concatenate([ft_all_z, ft_z])
            freq_all = np.concatenate([freq_all, freq])
            
            f_com.close()
            f_imu.close()
            file_count+=1

        # close the output file
        print("Finished processing")
        return ft_all_y, ft_all_z, freq_all


input_folders = ["data/raw/0405/", "data/raw/1605/", "data/raw/0106/", "data/raw/1506/"]

ft_y_comb, ft_z_comb, freq_comb = np.array([]), np.array([]), np.array([])
for day_folder in input_folders:
    transformer = FT(day_folder)
    ft_all_y, ft_all_z, freq_all = transformer.transformBags()

    #add the FT of one day to the combined
    ft_y_comb = np.concatenate([ft_y_comb, ft_all_y])
    ft_z_comb = np.concatenate([ft_z_comb, ft_all_z])
    freq_comb = np.concatenate([freq_comb, freq_all])

#saving the results to a file
np.savez("fourier_results.npz", ft_y_comb = ft_y_comb, ft_z_comb=ft_z_comb, freq_comb=freq_comb)

#plotting
fig, ax = plt.subplots(1, 2)
ax[0].plot(freq_comb, ft_y_comb)
ax[0].set_ylabel('Amplitude', fontsize = 12)
ax[0].set_xlabel('Frequency (Hz)', fontsize = 12)
#ax[0].set_ylim(-1000, 1000)

ax[1].plot(freq_comb, ft_z_comb)
ax[1].set_xlabel('Frequency (Hz)', fontsize = 12)
#ax[1].set_ylim(-1000, 1000)

plt.show()

# npzfile = np.load("calib_results3.npz")
# ret, mtx, dist, rvecs, tvecs = npzfile["ret"], npzfile["mtx"], npzfile["dist"], npzfile["rvecs"], npzfile["tvecs"]