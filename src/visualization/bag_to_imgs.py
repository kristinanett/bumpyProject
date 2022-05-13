#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Copyright 2016 Massachusetts Institute of Technology

"""Extract images from a rosbag.
"""

import os
import argparse
import sys

import cv2

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def main():
    """Extract a folder of images from a rosbag.
    """
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    #parser.add_argument("--output_dir", help="Output directory.", default = "cam" + input_rosbag)
    #parser.add_argument("--image_topic", help="Image topic.", default = "/cam1/image_rect/compressed")

    args = parser.parse_args()

    output_dir = "/".join(args.bag_file.split("/")[:-1]) + "/cam" + args.bag_file.split("/")[-1][:-4]
    image_topic = "/cam1/image_rect/compressed"

    print("Extract images from %s on topic %s into %s" % (args.bag_file,
                                                          image_topic, output_dir))

    #check output folder exists
    if os.path.isdir(output_dir):
        print("Output folder already exists")
        #sys.exit()
    else:
        os.mkdir(output_dir)
        print("New output directory created:", output_dir)                                            

    bag = rosbag.Bag(args.bag_file, "r")
    bridge = CvBridge()
    count = 0
    for topic, msg, t in bag.read_messages(topics=[image_topic]):
        #cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        cv_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8') #for compressed images

        cv2.imwrite(os.path.join(output_dir, "frame%06i.png" % count), cv_img)

        #check that it got written
        if not cv2.imwrite(os.path.join(output_dir, "frame%06i.png" % count), cv_img):
            #raise Exception("Could not write image")
            print("nope")
        if count%10 == 0:
            print("Wrote image %i" % count)
        count += 1

    print("Total written images: ", count)
    bag.close()

    return

if __name__ == '__main__':
    main()
