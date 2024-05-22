#! /usr/bin/env python3

from human_pose_estimation.Detection.DetectionModule import DetectionModule
from human_pose_estimation.Camera.CvL515Module import CvL515Module
from human_pose_estimation.Model.Yolov8Module import Yolov8Module
from human_pose_estimation.Visualization.CvVisualizationModule import CvVisualizationModule

import rospy
import rospkg
import os
import cv2

if __name__ == '__main__':

    rospy.init_node('human_pose_estimation_node')
    # Create CameraModule
    package_path = rospkg.RosPack().get_path('human_pose_estimation')
    
    file_path = package_path + "/data/recordings" + "/desk_standing2/"
    camera_path = package_path + "/camera/"

    camera = CvL515Module(file_path, camera_path)
    # Create ModelModule
    model = Yolov8Module()
    # Create VisuModule
    visu = CvVisualizationModule(model.kp_table)
    # Create DetectionModule

    detection = DetectionModule(camera, model)

    while not rospy.is_shutdown():
    # while True:

        image_rgb, image_depth = camera.getImages()

        skeletons2d = detection.detect("color_optical_frame", image_rgb)

        visu.publishKeypoint2D(image_rgb, skeletons2d)
        visu.publishSkeleton2D(image_rgb, skeletons2d)

        skeletons3d = detection.project(skeletons2d, image_depth)

        #visu.publishKeypoint3D(skeletons3d)
        visu.publishSkeleton3D(skeletons3d)

        # launch RosInputModule.run()
        #visu.visu2dSkel(skeleton)

        # masks = skeleton.applyMasks(images, 5)
        # clusters = clustering(masks, images)

        # visu.visu2dSkel(clusters)
    cv2.destroyAllWindows()