#! /usr/bin/env python3

import rospy

from human_pose_detection.Detection.DetectionModule import DetectionModule
from human_pose_detection.Camera.RosL515Module import RosL515Module
from human_pose_detection.Model.Yolov8Module import Yolov8Module
from human_pose_detection.Visualization.RosVisualizationModule import RosVisualizationModule

camera = None
detection = None
visu = None

def callback(frame_id, image_rgb, image_depth):
    if detection is not None:
        # Compute keypoints detection and 2d skeletons
        skeletons2d = detection.detect(frame_id, image_rgb)

        visu.publishKeypoint2D(image_rgb, skeletons2d)
        visu.publishSkeleton2D(image_rgb, skeletons2d)

        # Project 2d skeltons into 3d skeletons
        skeletons3d = detection.project(skeletons2d, image_depth)
        
        visu.publishKeypoint3D(skeletons3d)
        visu.publishSkeleton3D(skeletons3d)
    
if __name__ == '__main__':

    rospy.init_node('human_pose_estimation_node')
    # Create CameraModule
    camera = RosL515Module(callback)
    # Create ModelModule
    model = Yolov8Module(model_name = "yolov8x-pose-p6")
    # Create VisuModule
    visu = RosVisualizationModule(model.kp_table)
    # Create DetectionModule

    detection = DetectionModule(camera, model)

    rospy.spin()
        