#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import csv
import rosbag
from cv_bridge import CvBridge
import json
import argparse
import glob
import pandas as pd

#this code only works on folders of bag files
#this code assumes that every bag file has a corresponding mqtt file and there is exactly one mqtt log in each timestamped folder

class ProcessorClass():
    """
        Objective:
        Class to synchronize command and imu data with the images. Processed csv goes in data_processed folder

        Args:
            bagfile_dir (string): Directory with all the raw command and imu data to process
        """

    def __init__(self, dir):
        self.input_folder = dir

    def getCorrespondingCommand(self, time, f):
        current_line = json.loads(f.readline())
        current_dif = time-current_line["time"] 

        while True:
            new_line = json.loads(f.readline())
            new_dif = time-new_line["time"] 
            if abs(new_dif) < abs(current_dif):
                current_dif = new_dif
                current_line = new_line
            else:
                linear_vel = current_line["message"]["twist"]["linear"]["x"]
                angle = current_line["message"]["twist"]["angular"]["z"]
                com_time = current_line["time"]
                break
    
        return (com_time, linear_vel, angle)

    def getCorrespondingIMUdata(self, time, df):
        filtered_df = df.loc[(df['times'] > time-500) & (df['times'] < time+500)]
        sum = (filtered_df['gyro_y']**2 + filtered_df['gyro_z']**2).mean()
        return sum

    def processData(self):
        csv_file = "data/processed/data2.csv"
        image_topic = "/cam1/image_rect/compressed"

        img_file_list = sorted(glob.glob("data/processed/*.bag"))

        #finding the mqtt dir folder and imu folder using the first file name of the bag files
        parts = img_file_list[0].split("/")[-1].split("_")[1].split("-")
        date = parts[2] + parts[1]  # data as format 0405
        mqtt_folder_list = sorted(glob.glob("data/raw/" + date + "/mqtt/*/"))
        imu_file_list = sorted(glob.glob("data/raw/" + date + "/imu/*.txt"))

        # open the output file in write mode
        f_csv = open(csv_file, 'w', newline = "")
        header = ['lin1', 'lin2', 'lin3', 'lin4', 'lin5', 'lin6', 'lin7', 'lin8', 'ang1', 'ang2', 'ang3', 'ang4', 'ang5', 'ang6', 'ang7', 'ang8', 'imu1', 'imu2', 'imu3', 'imu4', 'imu5', 'imu6', 'imu7', 'imu8']
        writer = csv.writer(f_csv)
        writer.writerow(header)

        k=0
        for input_file, com_folder, imu_file in zip(img_file_list[:], mqtt_folder_list[:], imu_file_list[:]):
            print("Processing file:", input_file, "\t", k+1, "/", len(img_file_list))
            input_bag = rosbag.Bag(input_file, "r")
            f_com = open(com_folder + "dir/log000", "r")
            f_imu = open(imu_file, "r")
            df = pd.read_csv(f_imu, sep=" ", usecols=[0,2,3], names=["times","gyro_y", "gyro_z"])
            bridge = CvBridge()

            i=0
            for topic, msg, t in input_bag.read_messages(topics=[image_topic]):
                #print("image idx=", i, "time", t)
                lin_coms = []
                ang_coms = []
                imus = []
                img_time = t.to_nsec()/1000000

                #looping 8 times for each image to get 8 future commands and corresponding imu values
                for j in range(8):
                    #saving where the file pointer was after the 1st loop (t+1s)
                    if j == 1:
                        return_pos = f_com.tell()

                    #getting the commands and the imu and appending them to lists
                    com_time, lin, ang = self.getCorrespondingCommand(img_time+((j+1)*1000), f_com)
                    imu = self.getCorrespondingIMUdata(com_time, df)
                    lin_coms.append(lin)
                    ang_coms.append(ang)
                    imus.append(imu)

                #returning the pointer to where it was at t+1s 
                f_com.seek(return_pos)

                # write a row to the csv file
                writer.writerow(lin_coms+ang_coms+imus)
                i+=1

            print("Wrote", i, "lines to csv file")
            input_bag.close()
            f_com.close()
            f_imu.close()
            k+=1

        # close the output file
        f_csv.close()
