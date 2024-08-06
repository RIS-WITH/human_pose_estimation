#! /usr/bin/env python3

from human_pose_estimation.Camera.RosL515Module import RosL515Module

import rospy
import rospkg
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


package_path = rospkg.RosPack().get_path('human_pose_estimation')

def callback(frame_id, image_rgb, image_depth, cpt_image):
    # r = rospkg.RosPack()
    # package_path = r.get_path('human_pose_estimation')
    saving_dir = "/data/recordings/"

    name = package_path + saving_dir + folder_name + "/" + str(cpt_image)

    cv2.imwrite(name + "_rgb.png", image_rgb)
    cv2.imwrite(name + "_depth.png", image_depth.astype(np.uint16))


if __name__ == '__main__':

    rospy.init_node('human_pose_estimation_node')

    # package_path = rospkg.RosPack().get_path('human_pose_estimation')

    saving_dir = "/data/recordings/"
    folder_name = "stephy_t_rex" #stephy_t_rex" #"guillaume_t_rex"
    
    if(os.path.isdir(package_path + saving_dir + folder_name + "/") == False):
        creat_path = os.path.join(package_path + saving_dir, folder_name) 
        os.mkdir(creat_path) 
   
    camera = RosL515Module(callback)

    while not rospy.is_shutdown():
        
        rospy.spin()
        
    