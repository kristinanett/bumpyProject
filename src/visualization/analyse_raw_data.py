import glob
import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
import json

class Analyser():
    def __init__(self, dir):

        self.imu_folder = dir + 'imu/' #'data/raw/0405/imu/'
        self.mqtt_folder = dir + 'mqtt/' #'data/raw/0405/mqtt/'

        self.imu_file_list = sorted(glob.glob(self.imu_folder + "*.txt"))
        self.mqtt_folder_list = sorted(glob.glob(self.mqtt_folder + "*/"))

        self.df_imu = self.readIMUData()
        self.df_com = self.readCOMData()

        print(self.df_imu.shape) #(111108, 3)
        print(self.df_com.shape) #(27548, 3)


    def readIMUData(self):
        df_imu = pd.DataFrame()
        for idx, imu_file in enumerate(self.imu_file_list):

            #get dataframe
            f_imu = open(imu_file, "r")
            df_current_file = pd.read_csv(f_imu, sep=" ", usecols=[0,2,3], names=["times","gyro_y", "gyro_z"])
            df_current_file = df_current_file.dropna()

            #filtering out standing parts
            lines = open(self.mqtt_folder_list[idx] + "dir/log000", "r").readlines()
            com_start_time = json.loads(lines[0])["time"]/1000 #seconds
            com_end_time = json.loads(lines[-1])["time"]/1000 #seconds
            df_current_file = df_current_file.loc[(df_current_file['times'] > (com_start_time * (10 ** 3))+1000) & (df_current_file['times'] < ((com_end_time -  8.2) * (10 ** 3)))]
            
            df_imu = pd.concat([df_imu, df_current_file], ignore_index = True)
        return df_imu 

    def readCOMData(self):
        df_com = []
        for com_folder in self.mqtt_folder_list:
            f_com = open(com_folder + "dir/log000", "r")
            for line in f_com:
                current_line = json.loads(line)
                com_time = current_line["time"]
                linear_vel = current_line["message"]["twist"]["linear"]["x"]
                angle = current_line["message"]["twist"]["angular"]["z"]
                df_com.append([com_time, linear_vel, angle])
        return pd.DataFrame(df_com, columns=['time', "x", "z"])

analyser1 = Analyser("data/raw/0405/")
analyser2 = Analyser("data/raw/1605/")
analyser3 = Analyser("data/raw/0106/")


print(analyser1.df_imu.describe())
print()
print(analyser2.df_imu.describe())
print()
print(analyser3.df_imu.describe())

print(analyser1.df_imu[analyser1.df_imu > 40.0].count())
print(analyser2.df_imu[analyser2.df_imu > 40.0].count())
print(analyser3.df_imu[analyser3.df_imu > 40.0].count())
print(analyser3.df_imu[analyser3.df_imu['gyro_y'] > 40].index)

fig, axes = plt.subplots(1, 2)

analyser1.df_imu.hist("gyro_y", bins='auto', ax=axes[0])  
analyser2.df_imu.hist("gyro_y", bins='auto', ax=axes[0])
analyser3.df_imu.hist("gyro_y", bins='auto', ax=axes[0])
axes[0].set_title("IMU histogram gyro-y")

analyser1.df_imu.hist("gyro_z", bins='auto', ax=axes[1])  
analyser2.df_imu.hist("gyro_z", bins='auto', ax=axes[1])
analyser3.df_imu.hist("gyro_z", bins='auto', ax=axes[1])
axes[1].set_title("IMU histogram gyro-z")
plt.show()