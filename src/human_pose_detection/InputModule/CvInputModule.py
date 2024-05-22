import InputModule

import cv2
import rospkg
import matplotlib.pyplot as plt
import fnmatch

import os
import numpy as np 

class CvInputModule(InputModule):
    def __init__(self, camera, mode, model):
        super.__init__(self, camera, model)
        self.mode_ = mode

    # Changes the mode from pointcloud to depth image to get the z coordinates
    def setupMode(self):
        if(self.mode_ == "cloud"):
            self.readSyncRGBCloudFrames()
        elif(self.mode_ == "depth"):
           self.readRGBDepthFrames()
        else:
            print("mode not supported")

    # Handles the synchronized rgb and aligned depth topics
    def readRGBDepthFrames(self, filename, save_skeleton = False):

        if(os.path.isfile(self.package_path + "/" + filename) == True):
            print("single file")
        else:
            print("folder")
            num_img = len(fnmatch.filter(os.listdir(self.package_path + "/" + filename), '*.png'))

            for img_index in range(int(num_img/2)):
                name = filename + "/" + str(img_index)
                self.syncRGBDepthCallback(image_rgb = name + "_rgb.png", image_depth = name + "_depth.png", save_img= False)

        if(save_skeleton == True):
            self.save_skeletons(folder_name="skeletons", filename="skeletons_test")
    
    def syncRGBDepthCallback(self, image_rgb, image_depth, save_img = False):

        frame_id = image_rgb.header.frame_id

        # self.package_path + "/" + dir + "/" + filename +
        frame_rgb = cv2.imread(self.package_path + "/" + image_rgb, cv2.IMREAD_UNCHANGED)
        # === cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED required because implicit conversion from int16 to int8
        frame_depth = cv2.imread(self.package_path  + "/" + image_depth , cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)

        # ==================== Save synchronized images ==================
        if(save_img == True):
            self.save_image(frame_rgb, frame_depth, "test_folder")
       
        detected_persons_2d = self.model.predictDetections(frame_rgb, frame_id, 5, True)
        detected_persons_3d = self.model.projectDetectionsDepth(detected_persons_2d, frame_depth, True)

        # ==================== Save skeletons ==================
        self.skeletons_array_.append(detected_persons_3d)
        self.cpt_ += 1

    def save_image(self, frame_rgb, frame_depth, folder_name):
        r = rospkg.RosPack()
        package_path = r.get_path('human_pose_detection')

        if(os.path.isdir(package_path + "/" + folder_name + "/") == False):
            creat_path = os.path.join(package_path, folder_name) 
            os.mkdir(creat_path) 

        name = package_path + "/" + folder_name + "/" + str(self.cpt_)

        cv2.imwrite(name + "_rgb.png", frame_rgb)
        cv2.imwrite(name + "_depth.png", frame_depth.astype(np.uint16))


