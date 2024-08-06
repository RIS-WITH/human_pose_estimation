import rospkg

import cv2
import numpy as np
import pyrealsense2 as rs2

import fnmatch
import os

from ultralytics import YOLO

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import matplotlib.pyplot as plt

from human_pose_estimation.utils_camera import setup_intrinsic

from human_pose_estimation.utils_visu import displaySkeletons2D, displaySkeletons3D

from human_pose_estimation.src.human_pose_estimation.Skeleton.utils_human_pose import Keypoint2D, Keypoint3D, Skeleton2d, Skeleton3d, KeyPointIndexTableMediapipe, KeyPointIndexTableYolo

from sklearn.cluster import KMeans, MiniBatchKMeans, AffinityPropagation, MeanShift

def predict_single_image(dir = None, filename = None, color_int = None, visualize = False, fig = None, ax = None, model = None, keypoint_table = None):

    r = rospkg.RosPack()
    path = r.get_path('human_pose_estimation')

    image_rgb = cv2.imread(path + "/" + dir + "/" + filename + "_rgb.png", cv2.IMREAD_UNCHANGED)
    # === cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED required because implicit conversion from int16 to int8
    image_depth = cv2.imread(path + "/" + dir + "/" + filename + "_depth.png" , cv2.IMREAD_ANYDEPTH | cv2.IMREAD_UNCHANGED)

    # ========= Compute histogram ========
    # image_depth_list = image_depth[image_depth > 0]

    # hist = cv2.calcHist([image_depth_list], [0], None, [65536], [0, 65536])

    # fig2 = plt.figure()
    # plt.plot(hist)
    # plt.title("Grayscale Histogram")
    # plt.xlabel("Bins")
    # plt.ylabel("# of Pixels")
    # plt.xlim([0, 6000])

    # name = path + "/desk_standing2_hist/" + filename + "_hist_bis2.png"
    # fig2.savefig(name)
    
    detections = model.predict(image_rgb, show = False)
    dim_rgb = image_rgb.shape

    skeletons_2d, skeletons_3d = [], []

    # we have to take the first element for no reason
    kp_n = detections[0].keypoints.xyn
    conf_n = detections[0].keypoints.conf

    frame_id = "camera_link"

    if((kp_n is not None) & (conf_n is not None)):
        kp_n = kp_n.cpu().data.numpy()
        conf_n = conf_n.cpu().data.numpy()

        for i in range(0, len(kp_n)):
            skeletons_2d.append(computeDetectionKeypoints(kp_n[i], conf_n[i], dim_rgb, i, frame_id, keypoint_table))

        if(visualize == True):
            displaySkeletons2D(image_rgb, image_depth, skeletons_2d, keypoint_table)

        for skeleton in skeletons_2d:
            skeletons_3d.append(projectSkeletonDepth(skeleton, image_depth, color_int))

        if(visualize == True):
            displaySkeletons3D(skeletons_3d, keypoint_table, fig, ax)

def projectSkeletonDepth(detected_person, img_depth, color_intrinsic, mask = None):
# passer en argument la fonction de clustering
# passer en arg l'objet du masque et la bonne fonction est appelé pour appliquer le masque selon son type

    skeleton_3d = Skeleton3d(detected_person.skeleton_id_, detected_person.frame_id_)

    for keypoint in detected_person.keypoints:

        if(keypoint.x_ == 0 and keypoint.y_ == 0):
            new_kp = Keypoint3D(keypoint.label_, keypoint.confidence_, 0, 0, 0)
        else:
            if(mask == None):
                pix_x, pix_y = int(keypoint.x_), int(keypoint.y_)
                depth_scale = 0.001
                depth_value = img_depth[pix_y, pix_x]*depth_scale

                depth_point = rs2.rs2_deproject_pixel_to_point(color_intrinsic, [pix_x, pix_y], depth_value)
                new_kp = Keypoint3D(keypoint.label_, keypoint.confidence_, depth_point[0], depth_point[1], depth_point[2])
            else:
                # if(new_kp.is_null_ == True):
                print("Null le kp")
                step = 3
                dist_values = []  
                if((pix_y - step > 0) & (pix_x - step > 0)):
                    pixs_x, pixs_y = np.arange(pix_x-step, pix_x+step, 1, dtype=int), np.arange(pix_y-step, pix_y+step, 1, dtype=int)
                    for x_i in pixs_x:
                        for y_i in pixs_y :
                            # print([x_i, y_i])
                            depth_value = img_depth[y_i, x_i]*depth_scale
                            depth_point = rs2.rs2_deproject_pixel_to_point(color_intrinsic, [x_i, y_i], depth_value)
                            dist_values.append(depth_point[-1])
                            
                print("new values : ", dist_values)
                dist_values = np.array(dist_values)
                dist_values = dist_values.reshape(-1,1)
                kp_kmeans = KMeans(n_clusters=2, n_init = 'auto').fit(dist_values)

                # print("pixel value", img_depth[y_i, x_i]*depth_scale)
                data = np.array(img_depth[y_i, x_i]*depth_scale)
                # print(data)
                data = data.reshape(1,-1)
                kp_kmeans.predict(data)

            
            # plt.show() 


        skeleton_3d.addKeypoint(new_kp)
    return skeleton_3d

def computeDetectionKeypoints(kp_n, conf_n, dim_rgb, id, frame_id, table):

    skeleton_2d = Skeleton2d(id, frame_id)
    width, height, _ = dim_rgb

    for kp_index in range(0, len(kp_n)):
        label = table.getKeypointName(kp_index)
        kp = kp_n[kp_index]

        pix_x_rgb, pix_y_rgb  = int(height*kp[0]), int(width*kp[1])
        #print("label : ", label, " conf : ", conf_n[kp_index], " normalized : ", kp[0], kp[1], "pixel values :", pix_x_rgb, pix_y_rgb)

        if((pix_y_rgb < width) & (pix_x_rgb < height)):
            new_kp = Keypoint2D(label, conf_n[kp_index], pix_x_rgb, pix_y_rgb)
        else:
            new_kp = Keypoint2D(label, conf_n[kp_index], 0, 0)

        skeleton_2d.addKeypoint(new_kp)

    return skeleton_2d

def predict_images_folder(dir, fig, ax, visualize, model, keypoint_table, color_int):
    r = rospkg.RosPack()
    path = r.get_path('human_pose_estimation')

    num_img = len(fnmatch.filter(os.listdir(path + "/" + dir), '*.png'))

    for img_index in range(int(num_img/2)):
        print("IMG : ", str(img_index))
        predict_single_image(dir, str(img_index), color_int, visualize, fig, ax, model, keypoint_table)
        # plt.show()

def setup_model(model = None):

    r = rospkg.RosPack()
    path = r.get_path('human_pose_estimation')

    if(model == "yolo"):
        model = YOLO(model = path + "/models/yolo/" + "yolov8x-pose-p6.pt", verbose=False)
        table = KeyPointIndexTableYolo()
        
    elif(model == "mediapipe"):
        num_poses, detect_conf, pres_conf, track_conf  = 4, 0.8, 0.8, 0.8

        base_options = python.BaseOptions(model_asset_path= path + "/models/mediapipe/" +'/pose_landmarker_heavy.task')
        options = vision.PoseLandmarkerOptions(
            base_options=base_options, num_poses = num_poses, min_pose_detection_confidence = detect_conf, 
            min_pose_presence_confidence = pres_conf, min_tracking_confidence = track_conf, output_segmentation_masks=False)
        
        model = vision.PoseLandmarker.create_from_options(options)
        table = KeyPointIndexTableMediapipe()

    return [model, table]

def setup_camera(camera = None):

    r = rospkg.RosPack()
    path = r.get_path('human_pose_estimation')

    if(camera == "l515"):
        color_intrinsic = setup_intrinsic(path + "/" + str("color_intrinsic_"+camera) + ".npy")
        depth_intrinsic = setup_intrinsic(path + "/" + str("depth_intrinsic_"+camera) + ".npy")


    return [color_intrinsic, depth_intrinsic]

def predict_detections(mode = None, dir = None, filename = None, model = None, visualize = False):

    [model, table] = setup_model(model)
    [color_intrinsic, depth_intrinsic] = setup_camera("l515")

    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.view_init(elev=-90, azim=-90, roll=0)
    
    # plt.axis([-10, 10, -10, 10])

    if(mode == "single"):
        print("single mode detection")
        predict_single_image(model=model, keypoint_table= table, dir=dir, filename=filename, color_int=color_intrinsic, fig=fig, ax = ax, visualize=visualize)

    elif(mode == "folder"):
        print("folder mode detection") 
        predict_images_folder(dir=dir, model=model, keypoint_table = table, color_int=color_intrinsic, fig=fig, ax = ax, visualize=visualize)

    else:
        print("unrecognized mode detection")
