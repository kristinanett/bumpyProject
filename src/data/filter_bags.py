import rosbag
import argparse
import glob
import os
import json

#this code only works on folders of bag files
#this code assumes that every bag file has a corresponding mqtt file and there is exactly one mqtt log in each timestamped folder

class filterClass():
    def __init__(self, dir):
        
        """
        Objective:
        Class to filter out images from beginning and end of bagfile based on the com file start and end. Processed bagfiles go in data_processed folder

        Args:
            bagfile_dir (string): Directory with all the raw bagfiles with images to process
        """
        self.input_folder = dir + 'cam/'
        self.file_list = sorted(glob.glob(self.input_folder + "*.bag"))
        

    def filterBags(self):
        image_topic = "/cam1/image_rect/compressed"

        for input_file in self.file_list:

            #check if output file already exists
            output_file = "data/processed/" + input_file.split("/")[-1]
            if os.path.isfile(output_file):
                print("The filtered output file already exists. Overwriting", output_file)
                ans = input("Do you want to proceed? (y/n)")
                # Continue to the next file if the input value is 'n'
                if (ans.lower() == 'n'):
                    continue
            else:
                print("Processing bag file %s. Saving into %s" % (input_file, output_file))

            #getting the end time of the bag from its info (only used for constant cutoff)
            input_bag = rosbag.Bag(input_file, "r") 
            #info_dict = yaml.load(input_bag._get_yaml_info(), Loader=yaml.FullLoader)
            #end_time = info_dict["end"]

            #accessing the velocity command document to get the time that commands started sending and saving (to filter out all images taken before)
            file_idx = self.file_list.index(input_file)
            mqtt_folder_list = sorted(glob.glob("/".join(input_file.split("/")[:3]) + "/mqtt/*/"))
            mqtt_folder = mqtt_folder_list[file_idx]
            print("Corresponding MQTT folder", mqtt_folder)
            f = open(mqtt_folder + "dir/log000", "r")
            lines = f.readlines()
            com_start_time = json.loads(lines[0])["time"]/1000 #seconds
            com_end_time = json.loads(lines[-1])["time"]/1000 #seconds
    
            with rosbag.Bag(output_file, 'w') as output_bag:
                for topic, msg, t in input_bag.read_messages(topics=[image_topic]):
                    if t.to_nsec() > (com_start_time * (10 ** 9)) and t.to_nsec() < (com_end_time * (10 ** 9)) -  8.2 * (10 ** 9):
                        output_bag.write(topic, msg, t)
                    else:
                        pass

        print("Finished filtering")



