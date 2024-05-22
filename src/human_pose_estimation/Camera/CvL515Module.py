import numpy as np

import pyrealsense2 as rs2
import rospkg 

import os
import fnmatch
import cv2
# Camera fait le chargement des images / flux videos + setup callback

class CvL515Module():
    def __init__(self, filepath, camera_path):
        print("Camera module")

        self.color_intrinsic = None
        self.depth_intrinsic = None
        self.depth_to_color_extrinsic = None
        self.depth_scale = 0.001

        self.data_path = filepath
        self.cpt_img = 0

        self.setCameraParams(camera_path)

    def setCameraParams(self, camera_path):
        self.color_intrinsic = self.setup_intrinsic(camera_path + str("color_intrinsic_l515") + ".npy")
        self.depth_intrinsic = self.setup_intrinsic(camera_path + str("depth_intrinsic_l515") + ".npy")

    def getImages(self):
        if(os.path.isfile( self.data_path + "_rgb.png") == True):
            image_rgb_name = self.data_path + "_rgb.png"
            image_depth_name = self.data_path + "_depth.png"
        elif(os.path.isdir(self.data_path) == True):
            image_rgb_name = self.data_path + "/" + str(self.cpt_img) + "_rgb.png"
            image_depth_name = self.data_path + "/" + str(self.cpt_img) + "_depth.png"
            # Increase img counter
            self.cpt_img += 1

        frame_rgb = cv2.imread(image_rgb_name, cv2.IMREAD_UNCHANGED)
        # === cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED required because implicit conversion from int16 to int8
        frame_depth = cv2.imread(image_depth_name , cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)

        return [frame_rgb, frame_depth]
    
    def setup_intrinsic(self, intrinsic_path = None):

        # width, height, K[2], K[5], K[0], K[4], coeffs, dist_model
        intrinsic_array = np.load(intrinsic_path, allow_pickle= True)
        
        intrinsic = rs2.intrinsics()
        intrinsic.width = intrinsic_array[0]
        intrinsic.height = intrinsic_array[1]
        intrinsic.ppx = intrinsic_array[2]
        intrinsic.ppy = intrinsic_array[3]
        intrinsic.fx = intrinsic_array[4]
        intrinsic.fy = intrinsic_array[5]
        intrinsic.coeffs = intrinsic_array[6]
        if intrinsic_array[7] == 'plumb_bob':
            intrinsic.model = rs2.distortion.brown_conrady
        elif intrinsic_array[7] == 'equidistant':
            intrinsic.model = rs2.distortion.kannala_brandt4

        
        return intrinsic
