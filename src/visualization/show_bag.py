#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""Visualize images from a rosbag.
"""

import os
import argparse
import sys

import cv2

import rosbag
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

def main():
    """Visualize images from a rosbag with cv2
    """
    parser = argparse.ArgumentParser(description="Extract images from a ROS bag.")
    parser.add_argument("bag_file", help="Input ROS bag.")
    parser.add_argument("--save", help="Save the video as an mp4 file", default = "false")

    args = parser.parse_args()
    image_topic = "/cam1/image_rect/compressed"
    print("Showing images from %s on topic %s" % (args.bag_file, image_topic))                                         

    bag = rosbag.Bag(args.bag_file, "r")
    bridge = CvBridge()

    if args.save == "true":
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        outfile = "reports/figures/vids/" + args.bag_file.split("/")[-1][:-4] +'.mp4'
        out = cv2.VideoWriter(outfile, fourcc, 5.0, (1490, 732))
        print("Saving video file", outfile)

    count = 0
    for topic, msg, t in bag.read_messages(topics=[image_topic]):

        #extract image
        cv_img = bridge.compressed_imgmsg_to_cv2(msg, desired_encoding='bgr8') #for compressed images

        #write timestamp (in seconds) on the image
        cv2.putText(cv_img, str(round(t.to_nsec()/1000000000,3)), (0,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2, 2)

        #show the image
        cv2.imshow('cv_img', cv_img)

        #save the image to video file if required
        if args.save == "true":
            out.write(cv_img)

        #show next frame when k pressed and stop if q is pressed
        key = cv2.waitKey(0)
        while key not in [ord('q'), ord('k')]:
            key = cv2.waitKey(0)
        if key == ord('q'):
            break

        if count%10 == 0:
            print("Showed image %i" % count)
        count += 1

    print("Total showed images: ", count)
    bag.close()

    if args.save == "true":
        out.release()

    cv2.destroyAllWindows()

    return

if __name__ == '__main__':
    main()


