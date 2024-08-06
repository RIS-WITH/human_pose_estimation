#! /usr/bin/env python3

from human_pose_estimation.Detection.DetectionModule import DetectionModule
from human_pose_estimation.Camera.CvL515Module import CvL515Module
from human_pose_estimation.Model.Yolov8Module import Yolov8Module
from human_pose_estimation.Visualization.CvVisualizationModule import CvVisualizationModule
from human_pose_estimation.Visualization.RosVisualizationModule import RosVisualizationModule
from human_pose_estimation.Detection import SquareMask
from human_pose_estimation.Evaluation.EvaluationModule import EvaluationModule

import rospy
import rospkg
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt


if __name__ == '__main__':

    rospy.init_node('human_pose_estimation_node')
    # Create CameraModule
    package_path = rospkg.RosPack().get_path('human_pose_estimation')
    
    file_path = package_path + "/data/recordings" + "/guillaume_t_rex"
    camera_path = package_path + "/camera/"

    camera = CvL515Module(file_path, camera_path)
    # Create ModelModule
    model = Yolov8Module()
    # Create VisuModule
    n_col, n_lin = 2, 2
    #visu = CvVisualizationModule(model.kp_table, n_col, n_lin)
    visu = CvVisualizationModule(model.kp_table, n_col, n_lin)
    # Create EvaluationModule
    evaluation = EvaluationModule(model.kp_table)
    # visu_ros = RosVisualizationModule(model.kp_table)
    # Choice of the masking patter with shape and length
    masking_pattern = ['square', 5]
    clustering_method = "knn"
    # Create DetectionModule

    detection = DetectionModule(camera, model, masking_pattern)

    skeletons3d_array, skeletons3d_min_array = [], []

    cpt = 0

    while not rospy.is_shutdown():

        image_rgb, image_depth = camera.getImages()
        cpt += 1
        
        if image_rgb is None:
            break

        skeletons2d = detection.detect("camera_color_optical_frame", image_rgb)
        
        #visu.publishKeypoint2D(image_rgb, skeletons2d, fig_name='keypoints_2d')

        #visu.publishSkeleton2D(image_rgb, skeletons2d, fig_name='skeletons_2d')
        
        skeletons3d, skeletons3d_cluster = detection.project(skeletons2d, image_depth)
        # if(skeletons3d.shape[0] >= 1):
        skeletons3d_array.append(skeletons3d)
        # if(not skeletons3d):
        #     print("empty")
        # else:
        #     skeletons3d_array.append(skeletons3d[0])
    
        # for kp in skeletons3d_cluster[0].keypoints:
        #     print(kp)
        #     print(kp.candidates_)
            # skeletons3d_array.append(skeletons3d)

        # visu.publishKeypoint3D(skeletons3d, fig_name = 'keypoints_3d')
        # visu.publishSkeleton3D(skeletons3d)
        # Publish clusters for each skeleton detection
        #visu.publishClusters3D(skeletons3d_cluster)

        skeletons3d_min, skeletons3dmin_cluster = detection.project(skeletons2d, image_depth, True)
        #visu.publishKeypoint3D(skeletons3d, fig_name = 'keypoints_3d_min')
        #visu.publishSkeleton3D(skeletons3d, fig_name = 'keypoints_3d_min')
        # if(skeletons3d_min.shape[0] >= 1):
        skeletons3d_min_array.append(skeletons3d_min)

        # masks = skeleton.applyMasks(images, 5)
        # clusters = clustering(masks, images)

        # visu.visu2dSkel(clusters)
        # if(cpt == 1):
        #     #clustered_keypoints = detection.applyingClusteringMethod(skeletons3d_array, )
        #     break
    # cv2.destroyAllWindows()
    print("Shape before reshape : ", len(skeletons3d_array))
    reshaped_skeletons_3d = detection.reshapeSkeletonArray(skeletons3d_array)
    print("Shape after reshape : ", reshaped_skeletons_3d.shape)
    print(reshaped_skeletons_3d)
    # for detect in skeletons3d_array:
    #     if(detect):
    #         print("detection shape ", detect.shape)
    #         break
    # ============== Compute statistics over skeletons ============
    print("\nBefore min distance :")
    res_kp = evaluation.computeKeypointStatistics(reshaped_skeletons_3d)
    #evaluation.kp_table_.joint_table =  [['left_hip', 'left_knee'], ['right_hip', 'right_knee'],  ['left_hip', 'right_hip'], ['right_knee', 'right_ankle'], ['left_knee', 'left_ankle']]
    results_default, occurences_default = evaluation.computeJointStatistics(skeletons3d_array)
    evaluation.displayLimbLength(results_default[0], fig_name = 'keypoint_default', save_plot= True, filename="keypoints_guillaume_t_rex.png")

    
    print("\nAfter min distance :")
    res_kp_min = evaluation.computeKeypointStatistics(skeletons3d_min_array)
    #evaluation.kp_table_.joint_table =  [['left_hip', 'left_knee'], ['right_hip', 'right_knee'],  ['left_hip', 'right_hip'], ['right_knee', 'right_ankle'], ['left_knee', 'left_ankle']]
    results_min, occurences_min = evaluation.computeJointStatistics(skeletons3d_min_array)
    evaluation.displayLimbLength(results_min[0], fig_name = 'keypoint_default', save_plot= True, filename="keypoints_guillaume_t_rex_square_5.png")
    
    # evaluation.compareKeypointDetections(res_kp, res_kp_min)
    # evaluation.compareJointDetections(occurences_default, occurences_min)
    # plt.pause(10)