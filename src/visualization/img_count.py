import rosbag
import argparse
import glob
import os
import json
import yaml

#this code only works on folders of bag files

class FilterClass2():
    def __init__(self, dir):
    
        self.input_folder = dir + 'cam/'
        self.file_list = sorted(glob.glob(self.input_folder + "*.bag"))
        
    def countMessages(self):
        countBig = 0
        d=0
        image_topic = "/cam1/image_rect/compressed"

        for input_file in self.file_list:

            countSmall = 0

            input_bag = rosbag.Bag(input_file, "r") 

            #counting all messages
            info_dict = yaml.load(input_bag._get_yaml_info(), Loader=yaml.FullLoader)
            # nrOfMsg = info_dict["messages"]
            dur = info_dict['duration']
            # print("Bag file %s has %s images and duration %s" % (input_file, nrOfMsg, dur))
            # count += nrOfMsg
            d += dur

            #counting messages on specific topic only
            for topic, msg, t in input_bag.read_messages(topics=[image_topic]):
                countSmall += 1
            
            countBig += countSmall
            print("Bag file %s has %s images" % (input_file, countSmall))

        print("Finished. Total count is", countBig, "And total duration", d, "s")


filter1 = FilterClass2("data/raw/0405/")
filter2 = FilterClass2("data/raw/1605/")
filter3 = FilterClass2("data/raw/0106/")
filter4 = FilterClass2("data/raw/1506/")

filter1.countMessages()
filter2.countMessages()
filter3.countMessages()
filter4.countMessages()
#raw
#0405 Total count is 27614 And total duration 2759.7134279999996 (counting by topic 13804)
#1605 Total count is 48229 And total duration 2416.0241309999997 (counting by topic 12073)

#processed
#0405 Total count is 12483 And total duration 2493.2297549999994 (counting by topic 12483)

