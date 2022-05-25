import rosbag
import glob
import csv
import json
from cv_bridge import CvBridge
import pandas as pd
from sensor_msgs.msg import Image
import cv2
import os
import sys

#this code only works on a folder of raw data
#this code assumes that every bag file has a corresponding mqtt file and there is exactly one mqtt log in each timestamped folder

class ProcessDay():
    def __init__(self, dir):
        
        """
        Objective:
            1. filter out images from beginning and end of bagfile based on the com file start and end.
            2. find syncronized mqtt and imu values for each kept image to be saved in data/processed/data.csv
            3. convert all kept images to png images to be saved in data/processed/imgs

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

        self.output_img_path = "data/processed/imgs/"
        self.output_data_path = "data/processed/data.csv"

        #check if data.csv exists and there are already processed image files in imgs
        imgs = sorted(glob.glob(self.output_img_path + "*.png"))
        if os.path.isfile(self.output_data_path) and imgs:
            self.mode = "a" #append to existing
            self.img_start_idx = len(imgs)
            ans = input("Found existing processed files, appending. Do you want to proceed? (y/n)")
            # Stop code if the input value is 'n'
            if (ans.lower() == 'n'):
                sys.exit(0)
        else:
            self.mode = "w" #write to new
            self.img_start_idx = 0
            print("Did not find already existing processed files, creating new")


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

    def getOutputCSV(self):
        f_csv = open(self.output_data_path, self.mode, newline = "")
        header = ['lin1', 'lin2', 'lin3', 'lin4', 'lin5', 'lin6', 'lin7', 'lin8', 'ang1', 'ang2', 'ang3', 'ang4', 'ang5', 'ang6', 'ang7', 'ang8', 'imu1', 'imu2', 'imu3', 'imu4', 'imu5', 'imu6', 'imu7', 'imu8']
        writer = csv.writer(f_csv)
        if self.mode == "w":
            writer.writerow(header)
        return f_csv, writer

    def syncAndSaveCSV(self, writer):
        return

    def writeIMG(self, bridge, msg, id):
        cv_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8') #for compressed images
        cv2.imwrite(os.path.join(self.output_img_path, "frame%06i.png" % id), cv_img)

        #check that it got written
        if not cv2.imwrite(os.path.join(self.output_img_path, "frame%06i.png" % id), cv_img):
            print("Could not write image")

        return

    def processBags(self):

        f_csv, csv_writer = self.getOutputCSV()

        #looping over all the bag files in one day
        file_count=0
        totalLineCount = 0
        current_img_idx = self.img_start_idx
        for input_cam_file, com_folder, imu_file in zip(self.cam_file_list, self.mqtt_folder_list, self.imu_file_list):

            input_bag = rosbag.Bag(input_cam_file, "r")
            f_com = open(com_folder + "dir/log000", "r")
            f_imu = open(imu_file, "r")
            df = pd.read_csv(f_imu, sep=" ", usecols=[0,2,3], names=["times","gyro_y", "gyro_z"])
            df = df.dropna() #drop rows with nan values
            #df_new = df[pd.to_numeric(df['times'], errors='coerce').notnull()] #filter out some broken rows that have strings
            #df_new['times'] = df_new['times'].astype(int) #make sure all timesteps are ints

            bridge = CvBridge()

            #accessing the velocity command document to get the time that commands started/stopped sending and saving (to filter out all images taken before/after)
            file_idx = self.cam_file_list.index(input_cam_file)
            mqtt_folder = self.mqtt_folder_list[file_idx]
            print(file_count+1, "/", len(self.cam_file_list), ":", "Bag file", input_cam_file, "\tMQTT folder", mqtt_folder)

            lines = open(mqtt_folder + "dir/log000", "r").readlines()
            com_start_time = json.loads(lines[0])["time"]/1000 #seconds
            com_end_time = json.loads(lines[-1])["time"]/1000 #seconds

            #get imu file start time
            imu_start_time = df["times"][0]/1000 #seconds
            imu_end_time = df["times"][-1]/1000 #seconds
    
            #loop over the messages/images in one bag and check the timestamps to be correct (save mqtt and imu data for correct)
            msg_count=0
            for topic, msg, t in input_bag.read_messages(topics=[self.image_topic]):
                lin_coms = []
                ang_coms = []
                imus = []
                img_time = t.to_nsec()/1000000
                
                #the messages that should be saved reach here
                if (t.to_nsec() > (com_start_time * (10 ** 9))) and (t.to_nsec() < ((com_end_time -  8.2) * (10 ** 9))) and \
                    (t.to_nsec() > (imu_start_time* (10 ** 9))) and (t.to_nsec() < ((imu_end_time -  8.2) * (10 ** 9))):
                    
                    #saving the message as a png image 
                    self.writeIMG(bridge, msg, current_img_idx+msg_count)

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
                        
                        #nan investigation
                        if j ==0 and (current_img_idx+msg_count) == 21278:
                            print(imu, com_time) #nan 1652711464.726
                            filtered_df = df.loc[(df['times'] > com_time-500) & (df['times'] < com_time+500)]
                            sum = (filtered_df['gyro_y']**2 + filtered_df['gyro_z']**2).mean()
                            print(filtered_df['gyro_y'])
                            print(filtered_df['gyro_z'])
                            print(sum)


                    #returning the pointer to where it was at t+1s 
                    f_com.seek(return_pos)

                    # write a row to the csv file
                    csv_writer.writerow(lin_coms+ang_coms+imus)
                    msg_count+=1

                else:
                    pass

            print("Wrote", msg_count, "lines to csv file")
            totalLineCount += msg_count
            current_img_idx += msg_count
            input_bag.close()
            f_com.close()
            f_imu.close()
            file_count+=1

        # close the output file
        f_csv.close()
        print("Finished processing. Wrote a total of", totalLineCount, "lines to the csv file")

