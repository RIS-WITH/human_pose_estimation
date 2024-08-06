#! /usr/bin/env python3

import rospy

from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image, PointCloud2

from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point, PointStamped, PoseArray, Pose, Quaternion, Vector3

######

import message_filters
from cv_bridge import CvBridge, CvBridgeError
import cv2

import struct
import numpy as np

from ultralytics import YOLO

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import pyrealsense2 as rs2

from .Skeleton.utils_human_pose import KeyPointIndexTableMediapipe, KeyPointIndexTableYolo, Keypoint3D, Keypoint2D, Skeleton2d, Skeleton3d

import math
import os

import rospkg
import matplotlib.pyplot as plt

from PIL import Image as Img

from . utils_visu import SkeletonVisualizer

class ROSDetectionModule:
    def __init__(self, camera, mode):
        # print("ROSDetectionModule init")
        self.camera = camera
        self.bridge = CvBridge()
        self.mode_ = mode

        #Subscribers
        self.sub_rgb = message_filters.Subscriber("/camera/color/image_raw", Image) # frame : camera_color_optical_frame
        self.sub_depth = message_filters.Subscriber("/camera/depth/image_rect_raw", Image) # frame : camera_depth_optical_frame
        self.sub_aligned_depth = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image) #frame : camera_color_optical_frame
        self.sub_confidence = message_filters.Subscriber("/camera/confidence/image_rect_raw", Image) # frame : camera_confidence_optical_frame
        self.sub_pointcloud = message_filters.Subscriber("/camera/depth_registered/points", PointCloud2) # frame : camera_color_optical_frame
        
        #For saving purposes
        self.skeletons_array_ = []
        self.cpt_ = 0

    # Changes the mode from pointcloud to depth image to get the z coordinates
    def setupMode(self):
        if(self.mode_ == "cloud"):
            self.readSyncRGBCloudFrames()
        elif(self.mode_ == "depth"):
           self.readRGBDepthFrames()
        else:
            rospy.loginfo("mode not supported")

    # Callbacks
    def readSyncRGBCloudFrames(self):
        self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_rgb, self.sub_pointcloud], queue_size=1, slop=0.1)
        self.ts.registerCallback(self.syncRGBCloudCallback)
        rospy.spin()
    
    def syncRGBCloudCallback(self, image_rgb, cloud):

        frame_id = image_rgb.header.frame_id

        frame_rgb = self.bridge.imgmsg_to_cv2(image_rgb)

        detected_persons_2d = self.predictDetections(frame_rgb, frame_id, 5, True)
        
        detected_persons_3d = self.projectDetectionsCloud(detected_persons_2d, cloud, True)

        # Process the detected 3d skeletons
        # compute a np.array containing each skeletons
        # afterwards, process the array by joint with a plot of each joint size variations (size over time) and mean size, std, variance per joint


     # Handles the synchronized rgb and aligned depth topics
    
    def readRGBDepthFrames(self):
        # If rosbag recorded without power cord plugged in, latency between rgb and depth topics -> requires slop=1
        self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_rgb, self.sub_aligned_depth], queue_size=1, slop=0.05)
        self.ts.registerCallback(self.syncRGBDepthCallback)
        rospy.spin()

        # Saving process for the skeletons in np array
        # r = rospkg.RosPack()
        # path = r.get_path('human_pose_estimation')
        # np.save(path + '/skeletons_1h_desk_standing2s05.npy', np.array(self.skeletons_array_, dtype=object))
        # print("saved array")

    def syncRGBDepthCallback(self, image_rgb, image_depth):

        frame_id = image_rgb.header.frame_id

        frame_rgb = self.bridge.imgmsg_to_cv2(image_rgb, desired_encoding="bgr8")

        frame_depth = self.bridge.imgmsg_to_cv2(image_depth, desired_encoding="16UC1")
        print("Depth :", frame_depth.shape)

        #image_depth_flat = np.reshape(frame_depth, (921600))
        #img_depth_meter = 0.001*np.array(image_depth)
        #print("BINS : ", np.bincount(image_depth_flat))
        # frame_depth = np.array(frame_depth)
        # print(frame_depth.shape)
        frame_depth_list = frame_depth[frame_depth > 0]
        #print(frame_depth.shape)
        hist = cv2.calcHist([frame_depth_list], [0], None, [65536], [0, 65536])

        plt.figure()
        plt.title("Grayscale Histogram")
        plt.xlabel("Bins")
        plt.ylabel("# of Pixels")
        plt.plot(hist)
        plt.xlim([0, 6000])
       
        # plt.pause(5)

        #frame_depth = self.bridge.imgmsg_to_cv2(image_depth)

        # ==================== Save synchronized images ==================
        r = rospkg.RosPack()
        path = r.get_path('human_pose_estimation')

        name = path + "/desk_standing2_hist/" + str(self.cpt_)+ "_hist.png"
        plt.savefig(name)
        # print("RGB shape ", frame_rgb.shape)
        # print("Depth shape", frame_depth.shape)

        cv2.imwrite(path + "/desk_standing2/" + str(self.cpt_)+ "_rgb.png", frame_rgb)
        #cv2.imwrite(path + "/desk_standing2/" + str(self.cpt_)+ "_depth_bis.png", frame_depth.astype(np.uint16))
        cv2.imwrite(path + "/desk_standing2/" + str(self.cpt_)+ "_depth.png", frame_depth.astype(np.uint16))

        


        # depth_bis = cv2.imread(path + "/desk_standing2/" + str(self.cpt_)+ "_depth_bis.png", cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)
        # print("Depth bis", depth_bis.shape)
        # depth_bis_list = depth_bis[depth_bis > 0]
        # #print(frame_depth.shape)
        # hist = cv2.calcHist([depth_bis_list], [0], None, [65536], [0, 65536])

        # plt.figure()
        # plt.title("Grayscale Histogram BIS")
        # plt.xlabel("Bins")
        # plt.ylabel("# of Pixels")
        # plt.plot(hist)
        # plt.xlim([0, 6000])

        # name_bis = path + "/desk_standing2_hist/" + str(self.cpt_)+ "_hist_bis.png"
        # plt.savefig(name_bis)
        # ============================= Depth post processing via Realsense API ================================
        # color_map = rs2.colorizer()
        # rs_frame = rs2.frame.
        # color_map.set_option(RS2_OPTION_HISTOGRAM_EQUALIZATION_ENABLED, 1.f)
        # color_map.set_option(RS2_OPTION_COLOR_SCHEME, 2.f)

        # detected_persons_2d = self.predictDetections(frame_rgb, frame_id, 5, True)
        # detected_persons_3d = self.projectDetectionsDepth(detected_persons_2d, frame_depth, True)

        # ==================== Save skeletons ==================
        # self.skeletons_array_.append(detected_persons_3d)
        self.cpt_ += 1

    # Projection functions
    # def projectDetectionsCloud(self, detected_persons, cloud, visualize = False):

    #     skeletons_3d = []

    #     for detected_person in detected_persons:
    #         skeletons_3d.append(self.projectKeypoints3DCloud(detected_person, cloud))

    #     if(visualize == True):
    #         self.publishKeypoint3D(skeletons_3d)
    #         self.publishSkeleton3D(skeletons_3d)
    
    #     return skeletons_3d
    
    # def projectKeypoints3DCloud(self, detected_person, cloud):

    #     skeleton_3d = Skeleton3d(detected_person.skeleton_id_, detected_person.frame_id_)

    #     for i in range(0, len(detected_person.keypoints)):
    #         kp = detected_person.keypoints[i]

    #         index = kp.y_*cloud.row_step + kp.x_* cloud.point_step
    #         (x, y, z) = struct.unpack_from('fff', cloud.data, offset=index)

    #         if((math.isnan(x) == False and x != 0.0) & (math.isnan(y) == False and y != 0.0 ) & (math.isnan(z) == False and z != 0.0)):
    #             new_kp = Keypoint3D(kp.label_, kp.confidence_, x, y, z)
    #         else:
    #             new_kp = Keypoint3D(kp.label_, kp.confidence_, 0, 0, 0)
    #         skeleton_3d.addKeypoint(new_kp)
                

    #     # for kp in detected_person.keypoints:
    #     #     # index_row = np.multiply(int(kp.y_), cloud.row_step)
    #     #     # index_point = np.multiply(int(kp.x_), cloud.point_step)
    #     #     # index = index_row + index_point

    #     #     index = kp.y_*cloud.row_step + kp.x_* cloud.point_step
    #     #     (x, y, z) = struct.unpack_from('fff', cloud.data, offset=index)

    #     #     skeleton3d.addKeypoint(Keypoint3D(kp.label_, kp.confidence_,x, y, z))
        
    #     return skeleton_3d

    # def projectDetectionsDepth(self, detected_persons, img_depth, visualize = False):        
    #     skeletons_3d = []

    #     for detected_person in detected_persons:
    #         skeletons_3d.append(self.projectKeypoints3DDepth(detected_person, img_depth))
            
    #     if(visualize == True):
    #         self.publishKeypoint3D(skeletons_3d)
    #         self.publishSkeleton3D(skeletons_3d)

    #     return skeletons_3d
    
    # def projectKeypoints3DDepth(self, detected_person, img_depth):

    #     skeleton_3d = Skeleton3d(detected_person.skeleton_id_, detected_person.frame_id_)

    #     for keypoint in detected_person.keypoints:

    #         if(keypoint.x_ == 0 and keypoint.y_ == 0):
    #             new_kp = Keypoint3D(keypoint.label_, keypoint.confidence_, 0, 0, 0)
    #         else:
    #             pix_x, pix_y = int(keypoint.x_), int(keypoint.y_)
    #             depth_value = img_depth[pix_y, pix_x]*self.camera.depth_scale
            
    #             # color_intrinsic because aligned_depth to color image used / otherwise -> depth_intrinsic
    #             depth_point = rs2.rs2_deproject_pixel_to_point(self.camera.color_intrinsic, [pix_x, pix_y], depth_value)
    #             # print("img_depth pixel value :", pix_x, pix_y)
    #             # print("deprojected pixel value : ", depth_point)

    #             new_kp = Keypoint3D(keypoint.label_, keypoint.confidence_, depth_point[0], depth_point[1], depth_point[2])

                
    #             # step = 3
    #             # dist_values = []  
    #             # if((pix_y - step > 0) & (pix_x - step > 0)):
    #             #     pixs_x, pixs_y = np.arange(pix_x-step, pix_x+step, 1, dtype=int), np.arange(pix_y-step, pix_y+step, 1, dtype=int)
    #             #     for x_i in pixs_x:
    #             #         for y_i in pixs_y :
    #             #             # print([x_i, y_i])
    #             #             depth_point = rs2.rs2_deproject_pixel_to_point(self.camera.color_intrinsic, [x_i, y_i], depth_value)
    #             #             dist_values.append(depth_point)
    #             #             # if(depth_point[0] != 0):
    #             #             #     print("img_depth pixel value :", pix_x, pix_y)
    #             #             #     print("deprojected pixel value : ", depth_point)
    #             # print("old value : ", new_kp.z_)
    #             # print("new values : ", dist_values)

    #         skeleton_3d.addKeypoint(new_kp)
    #     return skeleton_3d
   

class MediapipeModule(ROSDetectionModule, SkeletonVisualizer):
    def __init__(self, camera, mode):
        SkeletonVisualizer.__init__(self)
        ROSDetectionModule.__init__(self, camera, mode)

        num_poses, detect_conf, pres_conf, track_conf  = 4, 0.8, 0.8, 0.8

        self.model = self.setup_detector(num_poses, detect_conf, pres_conf, track_conf)
        self.keypoint_model = KeyPointIndexTableMediapipe()

        #Publisher for 3D keypoints computed directly by mediapipe
        self.pub_world_kp = rospy.Publisher('/3D_world_keypoints', MarkerArray, queue_size=1)
        self.setupMode()

    def setup_detector(self, num_poses, detect_conf, pres_conf, track_conf):
        r = rospkg.RosPack()
        path = r.get_path('human_pose_estimation')

        base_options = python.BaseOptions(model_asset_path=path+'/pose_landmarker_heavy.task')
        options = vision.PoseLandmarkerOptions(
            base_options=base_options, num_poses = num_poses, min_pose_detection_confidence = detect_conf, 
            min_pose_presence_confidence = pres_conf, min_tracking_confidence = track_conf, output_segmentation_masks=False)
        
        model = vision.PoseLandmarker.create_from_options(options)

        return model

    def predictDetections(self, image_rgb, frame_id, step, visualize = False):

        v_thresh = 0.5
        skeletons_2d, skeletons_3d = [], []
        dim_rgb = image_rgb.shape

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        detections = self.model.detect(mp_image)

        #Normalized, World
        kp_n, kp_w = detections.pose_landmarks, detections.pose_world_landmarks

        if(len(kp_n) != 0):

            # 2D normalized coordinates
            for i in range(0, len(kp_n)):
                detected_2d_skeleton = self.computeDetectionKeypoints(kp_n[i], dim_rgb, i, frame_id, v_thresh)
                skeletons_2d.append(detected_2d_skeleton)

            # 3D world coordinates computed by mediapipe
            # for i in range(0, len(kp_w)):
            #     detected_3d_skeleton = self.computeDetection3DWorldKeypoints(kp_w[i], i, frame_id, v_thresh)
            #     skeletons_3d.append(detected_3d_skeleton)
        
            if(visualize == True):
                self.publishKeypoints2D(image_rgb, skeletons_2d, step)
                self.publishSkeleton2D(image_rgb, skeletons_2d)
                #self.publish3DWorldKeypoints(skeletons_3d)

        return skeletons_2d
    
    def computeDetectionKeypoints(self, kp_n, dim_rgb, id, frame_id, v_thresh):

        skeleton_2d = Skeleton2d(id, frame_id)
        width, height, _ = dim_rgb

        for kp_index in range(0, len(kp_n)):
            
            label = self.keypoint_model.getKeypointName(kp_index)
            kp = kp_n[kp_index]
            #kp.z is computed but not used here as we have the depth image for a more precise value
            # if(kp.x != 0 and kp.y != 0 and kp.z != 0 and kp.visibility > visibility_thresh and kp.presence > presence_thresh):
            pix_x_rgb, pix_y_rgb  = int(height*kp.x), int(width*kp.y)
            
            if((pix_y_rgb < width ) & (pix_x_rgb < height) & (kp.visibility > v_thresh)):
                new_kp = Keypoint2D(label, kp.visibility, pix_x_rgb, pix_y_rgb)
            else:
                new_kp = Keypoint2D(label, kp.visibility, 0, 0)

            skeleton_2d.addKeypoint(new_kp)
            
        return skeleton_2d
    
    # World keypoints computed by mediapipe
    def computeDetection3DWorldKeypoints(self, kp_w, id, frame_id, visibility_thresh):

        skeleton_3d = Skeleton3d(id, frame_id)

        for kp_index in range(0, len(kp_w)):
            
            label = self.keypoint_model.getKeypointName(kp_index)
            kp = kp_w[kp_index]
            # if(kp.x != 0 and kp.y != 0 and kp.z != 0 and kp.visibility > visibility_thresh and kp.presence > presence_thresh):
            new_kp = Keypoint3D(label, kp.visibility, kp.x, kp.y, kp.z)
            skeleton_3d.addKeypoint(new_kp)

        return skeleton_3d
    
    def publish3DWorldKeypoints(self, skeletons3d):
        # idea : have a different color for each detected person
        marker_array = MarkerArray()

        for skeleton in skeletons3d:

            marker = Marker()
            marker.header.frame_id = str(skeleton.frame_id_) #"camera_color_optical_frame"
            marker.id = int(skeleton.skeleton_id_)

            marker.type = Marker.POINTS
            marker.action = Marker.ADD

            marker.color = ColorRGBA(0, 1.0, 0.0, 1.0)
            marker.scale = Vector3(0.02, 0.02, 0.02)
            marker.pose.position = Point(0,0,0)
            marker.pose.orientation = Quaternion(0,0,0,1)

            for i in range(0, len(skeleton.keypoints)):
                kp = skeleton.keypoints[i]
                point = Point(kp.x_, kp.y_, kp.z_)
                marker.points.append(point)

            marker_array.markers.append(marker)

        self.pub_world_kp.publish(marker_array)

class YoloModule(ROSDetectionModule, SkeletonVisualizer):
    def __init__(self, camera, mode):
        SkeletonVisualizer.__init__(self)
        ROSDetectionModule.__init__(self, camera, mode)
        # passer en argument

        self.model = YOLO("yolov8x-pose-p6.pt")
        self.keypoint_model = KeyPointIndexTableYolo()

        self.setupMode()
    
    def predictDetections(self, image_rgb, frame_id, step, visualize = False):
        
        conf_thresh = 0.25
        detections = self.model.predict(image_rgb, conf = conf_thresh, show = False)
        dim_rgb = image_rgb.shape

        skeletons_2d = []

        # we have to take the first element for no reason
        kp_n = detections[0].keypoints.xyn
        conf_n = detections[0].keypoints.conf
        #print(kp_n)

        if((kp_n is not None) & (conf_n is not None)):
            kp_n = kp_n.cpu().data.numpy()
            conf_n = conf_n.cpu().data.numpy()
    
            for i in range(0, len(kp_n)):
                detected_2ds_skeleton = self.computeDetectionKeypoints(kp_n[i], conf_n[i], dim_rgb, i, frame_id)
                skeletons_2d.append(detected_2ds_skeleton)
        
            if(visualize == True):
                self.publishKeypoints2D(image_rgb, skeletons_2d, step)
                self.publishSkeleton2D(image_rgb, skeletons_2d)

        return skeletons_2d
    
    def computeDetectionKeypoints(self, kp_n, conf_n, dim_rgb, id, frame_id):

        skeleton_2d = Skeleton2d(id, frame_id)
        width, height, _ = dim_rgb

        for kp_index in range(0, len(kp_n)):
            label = self.keypoint_model.getKeypointName(kp_index)
            kp = kp_n[kp_index]
            # if(kp.all() != 0):
            
            pix_x_rgb, pix_y_rgb  = int(height*kp[0]), int(width*kp[1])
            #print("label : ", label, " conf : ", conf_n[kp_index], " normalized : ", kp[0], kp[1], "pixel values :", pix_x_rgb, pix_y_rgb)

            if((pix_y_rgb < width) & (pix_x_rgb < height)):
                new_kp = Keypoint2D(label, conf_n[kp_index], pix_x_rgb, pix_y_rgb)
            else:
                new_kp = Keypoint2D(label, conf_n[kp_index], 0, 0)

            #print(new_kp)
            skeleton_2d.addKeypoint(new_kp)

        return skeleton_2d




