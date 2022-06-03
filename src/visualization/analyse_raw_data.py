import glob
import numpy as np 
import pandas as pd
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
        for imu_file in self.imu_file_list:
            f_imu = open(imu_file, "r")
            df_current_file = pd.read_csv(f_imu, sep=" ", usecols=[0,2,3], names=["times","gyro_y", "gyro_z"])
            df_current_file = df_current_file.dropna()
            df_imu = pd.concat([df_imu, df_current_file])
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

analyser = Analyser("data/raw/0405/")
