import glob
import json
import numpy as np
import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import rosbag
import copy
import cv2
from cv_bridge import CvBridge
import pickle

plt.rcParams['axes.grid'] = True

#this code only works on a folder of raw data
#this code assumes that every bag file has a corresponding mqtt file and there is exactly one mqtt log in each timestamped folder

def getCorrespondingCommand(time, f):
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

def getCorrespondingIMUdata(time, df):
    filtered_df = df.loc[(df['times'] > time-500) & (df['times'] < time+500)]
    sum = (filtered_df['gyro_y']**2 + filtered_df['gyro_z']**2).mean()
    return sum, len(filtered_df)

def writeIMG(bridge, msg, id):
    output_img_path = "data/processed/ft"
    cv_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8') #for compressed images
    cv2.imwrite(os.path.join(output_img_path, "frame%06i.png" % id), cv_img)

    #check that it got written
    if not cv2.imwrite(os.path.join(output_img_path, "frame%06i.png" % id), cv_img):
        print("Could not write image")

    return

def transformBag(dir, input_cam_file, idxs, output_img_path):
    image_topic = "/cam1/image_rect/compressed"
    bridge = CvBridge()

    cam_folder = dir + 'cam/' #'data/raw/0405/cam/'
    imu_folder = dir + 'imu/' #'data/raw/0405/imu/'
    mqtt_folder = dir + 'mqtt/' #'data/raw/0405/mqtt/'

    input_cam_file = cam_folder + input_cam_file

    cam_file_list = sorted(glob.glob(cam_folder + "*.bag"))
    imu_file_list = sorted(glob.glob(imu_folder + "*.txt"))
    mqtt_folder_list = sorted(glob.glob(mqtt_folder + "*/"))

    #getting the imu and com files that correspond to the cam file
    file_idx = cam_file_list.index(input_cam_file)
    input_bag = rosbag.Bag(input_cam_file, "r")
    com_folder = mqtt_folder_list[file_idx]
    imu_file = imu_file_list[file_idx]
    print("Bag file", input_cam_file, "\tMQTT folder", com_folder)

    f_com = open(com_folder + "dir/log000", "r")
    f_imu = open(imu_file, "r")
    df = pd.read_csv(f_imu, sep=" ", usecols=[0,2,3], names=["times","gyro_y", "gyro_z"])
    df = df.dropna() #drop rows with nan values

    #accessing the velocity command document to get the time that commands started/stopped sending and saving (to filter out all images taken before/after)
    lines = open(com_folder + "dir/log000", "r").readlines()
    com_start_time = json.loads(lines[0])["time"]/1000 #seconds
    com_end_time = json.loads(lines[-1])["time"]/1000 #seconds

    #filtering out the beginning and end of file from imu data
    df_filtered = df.loc[(df['times'] > (com_start_time * (10 ** 3))+1000) & (df['times'] > (df['times'][0]+1000)) & (df['times'] < ((com_end_time -  8.2) * (10 ** 3)))]

    img_count = 0
    #filering the imu data by start and end image indexs (to get only a part of the whole file where it is on grass for example)
    for topic, msg, t in input_bag.read_messages(topics=[image_topic]):
        lin_coms = []
        ang_coms = []
        imus = []
        imus_samples = []
        img_time = t.to_nsec()/1000000

        if (t.to_nsec() > ((com_start_time + 1) * (10 ** 9))) and (t.to_nsec() > ((df["times"][0] + 1000) * (10 ** 6))) and (t.to_nsec() < ((com_end_time -  8.2) * (10 ** 9))):
            
            #getting the current* imu value (*averaged over the past second)
            averaged_df = df.loc[(df['times'] > (t.to_nsec()*10**(-6))-1000) & (df['times'] < (t.to_nsec()*10**(-6)))]
            
            #looping 8 times for each image to get 8 future commands and corresponding imu values
            for j in range(8):
                #saving where the file pointer was after the 1st loop (t+1s)
                if j == 1:
                    return_pos = f_com.tell()

                #getting the commands and the imu and appending them to lists
                com_time, lin, ang = getCorrespondingCommand(img_time+((j+1)*1000), f_com)
                imu, samples = getCorrespondingIMUdata(com_time, df)
                lin_coms.append(lin)
                ang_coms.append(ang)
                imus.append(imu)
                imus_samples.append(samples)

            #returning the pointer to where it was at t+1s 
            f_com.seek(return_pos)

            #check that the number of samples imu is averaging over is between 20 and 60
            #and that there are imu values in the last second (to obtain a not nan current imu value)
            if all(i > 20 and i < 60 for i in imus_samples) and len(averaged_df)>0:
                # if img_count > idxs[0] and img_count < idxs[1]:
                    #save image
                    # if not os.path.exists(output_img_path):
                    #     os.makedirs(output_img_path)
                    # cv_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8') #for compressed images
                    # cv2.imwrite(os.path.join(output_img_path, "frame%06i.png" % img_count), cv_img)
                if img_count == idxs[0]:
                    start_img_time = t.to_nsec() * (10 ** (-6)) #ms
                    # print("Start img time", start_img_time)
                elif img_count == idxs[1]:
                    end_img_time = t.to_nsec() * (10 ** (-6)) #ms
                    # print("End img time", end_img_time)
                img_count +=1
            else:
                print("Skipping a row and img - too few samples used: ", imus_samples)
            
            
    # print("Length before", len(df_filtered))
    df_filtered = df_filtered.loc[(df_filtered['times'] > (start_img_time)) & (df_filtered['times'] < (end_img_time))]
    print("IMU legnth after filtering", len(df_filtered))

    #take Fourier transform of filtered imu data
    ft_y = np.fft.rfft(np.array(df_filtered["gyro_y"]))
    ft_z = np.fft.rfft(np.array(df_filtered["gyro_z"]))
    freq = np.fft.rfftfreq(df_filtered.shape[0], d=0.025)
    
    f_com.close()
    f_imu.close()

    # close the output file
    return ft_y, ft_z, freq

#########################################TAKING THE FOURIER TRANSFORM AND SAVING TO FILE #######################################################

# input_folders = ["data/raw/0405/", "data/raw/1605/", "data/raw/0106/", "data/raw/1506/"]
# asphalt_files = ["_2022-05-04-10-15-15_0.bag", "_2022-05-16-14-48-48_0.bag", "_2022-06-01-10-26-24_0.bag", "_2022-06-15-09-50-57_0.bag"]
# asphalt_idxs = [(0, 170), (1, 114), (364, 551), (286, 843)]

# smallgrass_files = ["_2022-05-04-10-00-22_0.bag", "_2022-05-16-13-45-07_0.bag", "_2022-06-01-10-26-24_0.bag", "_2022-06-15-09-56-32_0.bag"]
# smallgrass_idxs = [(300, 1365), (3, 336), (764, 1265), (0, 386)]

# biggrass_files = ["_2022-05-04-13-06-44_0.bag", "_2022-05-16-14-10-40_0.bag", "_2022-06-01-11-07-54_0.bag", "_2022-06-15-10-14-06_0.bag"]
# biggrass_idxs = [(191, 477), (1, 167), (0, 294), (86, 241)]

# files = [asphalt_files, smallgrass_files, biggrass_files]
# all_idxs = [asphalt_idxs, smallgrass_idxs, biggrass_idxs]

# #Finding the Fourier transforms of the imu data in a specific image index range
# ft_y_all = [[[], [], [], []], [[], [], [], []], [[], [], [], []]]
# ft_z_all, freq_all = copy.deepcopy(ft_y_all), copy.deepcopy(ft_y_all)

# for i, (terrain_files, terrain_idxs) in enumerate(zip(files, all_idxs)):
#     for j, (day_folder, cam_file, idxs) in enumerate(zip(input_folders, terrain_files, terrain_idxs)):
#         #Keeping track of progress
#         if i == 0:
#             print("Starting asphalt:", j)
#         elif i == 1:
#             print("Starting small grass:", j)
#         elif i == 2:
#             print("Starting big grass:", j)
        
#         ft_y, ft_z, freq = transformBag(day_folder, cam_file, idxs, "data/processed/ft/ft"+(str(i)+str(j)))
#         ft_y_all[i][j] = ft_y
#         ft_z_all[i][j] = ft_z
#         freq_all[i][j] = freq

# #Fourier transform of cobblestone data separately
# ft_y_cob, ft_z_cob, freq_cob = transformBag("data/raw/1605/", "_2022-05-16-14-03-44_0.bag", (1, 73), "data/processed/ft/cobble")

# #saving the results to a file
# saved_data = dict(
#                   ft_y_all =ft_y_all, 
#                   ft_z_all=ft_z_all, 
#                   freq_all=freq_all, 
#                   ft_y_cob=ft_y_cob,
#                   ft_z_cob=ft_z_cob,
#                   freq_cob=freq_cob)

# with open('data/processed/additional/ft_results.pickle', 'wb') as outfile:
#     pickle.dump(saved_data, outfile, protocol=pickle.HIGHEST_PROTOCOL)

#############################################################################################################################################

#reading the results from a file for plotting
with open("data/processed/additional/ft_results.pickle", "rb") as input_file:
    data = pickle.load(input_file)

ft_y_all = data["ft_y_all"] 
ft_z_all = data["ft_z_all"]
freq_all = data["freq_all"]
ft_y_cob = data["ft_y_cob"]
ft_z_cob = data["ft_z_cob"]
freq_cob = data["freq_cob"]

# plotting 1 x 4 graphs (1 example of each of the 4 terrain types(includes cobble))
fig, ax = plt.subplots(1, 4)
for i in range(3):
    if i == 0:
        ax[i].set_ylabel('Amplitude', fontsize = 12)
    ax[i].plot(freq_all[i][0], ft_y_all[i][0].real)
    ax[i].set_xlabel('Frequency', fontsize = 12)
ax[3].plot(freq_cob, ft_y_cob.real)
ax[3].set_xlabel('Frequency', fontsize = 12)
plt.show()

#asphalt and cobble comparison graph
fig, ax = plt.subplots(1, 2)
ax[0].plot(freq_all[0][0], ft_y_all[0][0].real)
ax[0].set_ylabel('Amplitude', fontsize = 12)
ax[0].set_xlabel('Frequency', fontsize = 12)
ax[0].set_ylim(-500, 500)

ax[1].plot(freq_cob, ft_y_cob.real)
ax[1].set_xlabel('Frequency', fontsize = 12)
ax[1].set_ylim(-500, 500)
plt.show()

# plotting the 3 x 4 grid of graphs (4 examples for 3 terrain types)
rows=len(ft_y_all) #3
columns=len(ft_y_all[0]) #4
fig, ax = plt.subplots(rows, columns)
for i in range(rows):
    for j in range(columns):
        ax[i, j].plot(freq_all[i][j], ft_y_all[i][j].real)
        if j == 0:
            ax[i, j].set_ylabel('Amplitude', fontsize = 12)
            if  i ==2:
                ax[i, j].set_xlabel('Frequency', fontsize = 12)
        elif i == 2:
            ax[i, j].set_xlabel('Frequency', fontsize = 12)

plt.show()
