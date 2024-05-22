import rospy 
import os
import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import CameraInfo

from realsense2_camera.msg import Extrinsics

import pyrealsense2 as rs2

import numpy as np
import rospkg
import message_filters

from sensor_msgs.msg import Image, PointCloud2


class RosL515Module():
    def __init__(self, callback):
        self.bridge = CvBridge()
        self.color_intrinsic = None
        self.depth_intrinsic = None
        self.depth_to_color_extrinsic = None
        self.depth_scale = 0.001

        #Subscribers
        self.sub_rgb = message_filters.Subscriber("/camera/color/image_raw", Image) # frame : camera_color_optical_frame
        self.sub_depth = message_filters.Subscriber("/camera/depth/image_rect_raw", Image) # frame : camera_depth_optical_frame
        self.sub_aligned_depth = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image) #frame : camera_color_optical_frame
        self.sub_confidence = message_filters.Subscriber("/camera/confidence/image_rect_raw", Image) # frame : camera_confidence_optical_frame
        self.sub_pointcloud = message_filters.Subscriber("/camera/depth_registered/points", PointCloud2) # frame : camera_color_optical_frame

        self.running_ = False
        self.cpt_img = 0

        self.user_callback = callback
        self.setCameraParams()

        self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_rgb, self.sub_aligned_depth], queue_size=1, slop=0.05)
        self.ts.registerCallback(self.syncRGBDepthCallback)
        
    def setCameraParams(self):
        color_info = rospy.wait_for_message("/camera/color/camera_info", CameraInfo, timeout=5)
        self.setCameraInfo(color_info)
        depth_info = rospy.wait_for_message("/camera/depth/camera_info", CameraInfo, timeout=5)
        self.setCameraInfo(depth_info)
        depth_to_color_info = rospy.wait_for_message("/camera/extrinsics/depth_to_color", Extrinsics, timeout=5)
        self.setExtrinsicDepth2RGB(depth_to_color_info)

    def setCameraInfo(self, cameraInfo, save_camera = False):
        if((cameraInfo.header.frame_id == "camera_color_optical_frame") & (self.color_intrinsic == None)):
            self.color_intrinsic = rs2.intrinsics()
            self.color_intrinsic.width = cameraInfo.width
            self.color_intrinsic.height = cameraInfo.height
            self.color_intrinsic.ppx = cameraInfo.K[2]
            self.color_intrinsic.ppy = cameraInfo.K[5]
            self.color_intrinsic.fx = cameraInfo.K[0]
            self.color_intrinsic.fy = cameraInfo.K[4]

            if cameraInfo.distortion_model == 'plumb_bob':
                self.color_intrinsic.model = rs2.distortion.brown_conrady
            elif cameraInfo.distortion_model == 'equidistant':
                self.color_intrinsic.model = rs2.distortion.kannala_brandt4
            self.color_intrinsic.coeffs = [i for i in cameraInfo.D]
            rospy.loginfo("Color Intrinsic received")

            if(save_camera == True):
                self.save_camera_infos("color", cameraInfo)

        elif((cameraInfo.header.frame_id == "camera_depth_optical_frame") & (self.depth_intrinsic == None)):
            self.depth_intrinsic = rs2.intrinsics()
            self.depth_intrinsic.width = cameraInfo.width
            self.depth_intrinsic.height = cameraInfo.height
            self.depth_intrinsic.ppx = cameraInfo.K[2]
            self.depth_intrinsic.ppy = cameraInfo.K[5]
            self.depth_intrinsic.fx = cameraInfo.K[0]
            self.depth_intrinsic.fy = cameraInfo.K[4]

            if cameraInfo.distortion_model == 'plumb_bob':
                self.depth_intrinsic.model = rs2.distortion.brown_conrady
            elif cameraInfo.distortion_model == 'equidistant':
                self.depth_intrinsic.model = rs2.distortion.kannala_brandt4
            self.depth_intrinsic.coeffs = [i for i in cameraInfo.D]
            rospy.loginfo("Depth Intrinsic received")

            if(save_camera == True):
                self.save_camera_infos("depth", cameraInfo)
        else:
            rospy.loginfo("Unknown frame_id")

    def setExtrinsicDepth2RGB(self, extrinsic):
        if((extrinsic.header.frame_id == "depth_to_color_extrinsics") & (self.depth_to_color_extrinsic == None)):
            self.depth_to_color_extrinsic = rs2.extrinsics
            self.depth_to_color_extrinsic.rotation = extrinsic.rotation
            self.depth_to_color_extrinsic.translation = extrinsic.translation
            rospy.loginfo("Depth to Color Extrinsic received")

    def save_camera_infos(self, name, cameraInfo):
        #width, height, K[2], K[5], K[0], K[4], coeffs, dist_model
        if(name == "color"):
            intr = self.color_intrinsic
        elif(name == "depth"):
            intr = self.depth_intrinsic
        else:
            print("not recognized")

        intr_array = [intr.width, intr.height, intr.ppx, intr.ppy, intr.fx, intr.fy, [i for i in cameraInfo.D]]
        
        if cameraInfo.distortion_model == 'plumb_bob':
            intr_array.append("plumb_bob")
        elif cameraInfo.distortion_model == 'equidistant':
            intr_array.append("equidistant")

        r = rospkg.RosPack()
        path = r.get_path('human_pose_detection')
        np.save(path + '/' + name + '_intrinsic_l515.npy', np.array(intr_array))
        print("saved array")
    
    def save_image(self, frame_rgb, frame_depth, folder_name):
        r = rospkg.RosPack()
        package_path = r.get_path('human_pose_detection')

        if(os.path.isdir(package_path + "/" + folder_name + "/") == False):
            creat_path = os.path.join(package_path, folder_name) 
            os.mkdir(creat_path) 

        name = package_path + "/" + folder_name + "/" + str(self.cpt_)

        cv2.imwrite(name + "_rgb.png", frame_rgb)
        cv2.imwrite(name + "_depth.png", frame_depth.astype(np.uint16))

    def syncRGBDepthCallback(self, image_rgb, image_depth, save_img = False):

        frame_id = image_rgb.header.frame_id

        frame_rgb = self.bridge.imgmsg_to_cv2(image_rgb, desired_encoding="bgr8")
        frame_depth = self.bridge.imgmsg_to_cv2(image_depth, desired_encoding="16UC1")

        if(save_img == True):
            self.save_image(frame_rgb, frame_depth)
        
        self.user_callback(frame_id, frame_rgb, frame_depth)
        self.cpt_img += 1

        
