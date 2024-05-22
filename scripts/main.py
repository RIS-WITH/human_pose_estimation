#! /usr/bin/env python3

import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, CameraInfo, PointCloud2
from geometry_msgs.msg import Point, PointStamped, PoseArray, Pose, Quaternion, Vector3
from std_msgs.msg import ColorRGBA
from visualization_msgs.msg import Marker, MarkerArray
from realsense2_camera.msg import Extrinsics
import message_filters
import struct

from ultralytics import YOLO
import cv2
import numpy as np

import rospkg
import matplotlib.pyplot as plt
from PIL import Image

from mpl_toolkits.mplot3d.art3d import Line3D

import pyrealsense2 as rs2

from human_pose_estimation.utils_camera import L515_module
from human_pose_estimation.utils_yolo import YoloModule, MediapipeModule, KeyPointIndexTableYolo
from human_pose_estimation.utils_human_pose import Skeleton2d, Skeleton3d, Keypoint2D,  Keypoint3D

from human_pose_estimation.utils_detection import predict_detections, predict_images_folder

import os
import fnmatch
import time
# Yolo-m = (640*384)
# Yolo-x = (640*384)
# Yolo-x-p6 = (1280*768)

#access to pixels in pointcloud : pixel (i, j) using v[j * width() +i])

# YOLOv8 max input size : (480, 640, 3)
# Launch the rgb camera with this shape
#topics :  /camera/aligned_depth_to_color/image_raw /camera/color/image_raw /camera/aligned_depth_to_color/camera_info /camera/color/camera_info /camera/extrinsics/depth_to_color /camera/depth/camera_info



# record topics : rosbag record -O name.bag -a -x "(.*)theora(.*)|(.*)compressed(.*)"


if __name__ == '__main__':

    rospy.init_node('human_pose_estimation_node')

    r = rospkg.RosPack()
    path = r.get_path('human_pose_estimation')

    predict_detections(mode = "single", dir = "desk_standing2", filename = "29", model = "yolo", visualize = True)

   ### Load camera ###
   ### Load model ### 
   ### predict ###
   ### visu ###
   ### cluster ###
   ### visu ###
   ### metrics ###

    # ====================== Compute statistics ==========================
    # skeletons = np.load(path + '/skeletons_1h_desk_standing2.npy', allow_pickle=True)
    # table = KeyPointIndexTableYolo()
    # stats = compute_stat(skeletons, table, True)

    # stats = np.array(stats)
    # print(stats.shape)

    #     plt.plot(np.array(res_dist[:][i]).squeeze(), label= table.joint_table[i])

    # plt.plot(np.array(res_dist).squeeze())
  
    # for i in range(len(skeletons)):
    #     print("============== timestep ", i, " ==============")
    #     print(" nb detections : ", len(skeletons[i]))
    #     for j in range(len(skeletons[i])):
    #         print("============== detection ", j, " ==============")
    #         print(skeletons[i][j])
            #vis_3d_keypoints(skeletons[i][j].keypoints)
    
    # ======================== Process of skeleton computation ===============
    # camera = L515_module()
    # rospy.loginfo("L515_module created")
    
    # yolo = YoloModule(camera, "depth")
    # rospy.loginfo("YoloModule created")