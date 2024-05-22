import os
import rospkg
import numpy as np

import pyrealsense2 as rs2
import math
import struct

from human_pose_detection.Skeleton.utils_human_pose import Keypoint3D, Skeleton3d


class DetectionModule():
    def __init__(self, camera, model):
        self.camera_ = camera
        self.model_ = model
        self.package_path = rospkg.RosPack().get_path('human_pose_detection')

        #For saving purposes
        self.skeletons_array_ = []
        self.cpt_ = 0

    def save_skeletons(self, folder_name, filename):
        # r = rospkg.RosPack()
        # package_path = r.get_path('human_pose_detection')

        if(os.path.isdir(self.package_path + "/" + folder_name + "/") == False):
            creat_path = os.path.join(self.package_path, folder_name) 
            os.mkdir(creat_path) 

        name = self.package_path + "/" + folder_name + "/" + filename
        np.save(name + '.npy', np.array(self.skeletons_array_, dtype=object))
    
    def detect(self, frame_id, frame_rgb):

        detections = self.model_.predictDetections(frame_rgb, frame_id)
        
        return detections
    
    def project(self, skeletons2d, frame_depth):

        projections = self.projectDetectionsDepth(skeletons2d, frame_depth)

        return projections
    
    def projectKeypoints3DDepth(self, detected_person, img_depth):

        skeleton_3d = Skeleton3d(detected_person.skeleton_id_, detected_person.frame_id_)

        for keypoint in detected_person.keypoints:

            if(keypoint.x_ == 0 and keypoint.y_ == 0):
                new_kp = Keypoint3D(keypoint.label_, keypoint.confidence_, 0, 0, 0)
            else:
                pix_x, pix_y = int(keypoint.x_), int(keypoint.y_)
                depth_value = img_depth[pix_y, pix_x]*self.camera_.depth_scale
            
                # color_intrinsic because aligned_depth to color image used / otherwise -> depth_intrinsic
                depth_point = rs2.rs2_deproject_pixel_to_point(self.camera_.color_intrinsic, [pix_x, pix_y], depth_value)

                new_kp = Keypoint3D(keypoint.label_, keypoint.confidence_, depth_point[0], depth_point[1], depth_point[2])
                print("init value :", new_kp.z_)

                # Max 3 clusters : (front, middle, background)
                step = 10
                dist_values = []  
                if((pix_y - step > 0) & (pix_x - step > 0)):
                    pixs_x, pixs_y = np.arange(pix_x-step, pix_x+step, 1, dtype=int), np.arange(pix_y-step, pix_y+step, 1, dtype=int)
                    for x_i in pixs_x:
                        for y_i in pixs_y :
                            # print([x_i, y_i])
                            depth_point = rs2.rs2_deproject_pixel_to_point(self.camera_.color_intrinsic, [x_i, y_i], depth_value)
                            if(depth_point[2] > 0.25 and depth_point[2] < 9.0 ):
                                dist_values.append(depth_point[2])
                            # if(depth_point[0] != 0):
                            #     print("img_depth pixel value :", pix_x, pix_y)
                            #     print("deprojected pixel value : ", depth_point)
                if(len(dist_values) > 0 ):
                    #print("new values : ", dist_values)
                    print("min value :", np.min(dist_values))
                    print("mean value :", np.mean(dist_values))
                    print("max value :", np.max(dist_values))
            
            skeleton_3d.addKeypoint(new_kp)
        return skeleton_3d

    def projectDetectionsDepth(self, detected_persons, img_depth):        
        skeletons_3d = []

        for detected_person in detected_persons:
            skeletons_3d.append(self.projectKeypoints3DDepth(detected_person, img_depth))

        return skeletons_3d
    
    # ================ Cloud versions ---- Unused
    # def projectDetectionsCloud(self, detected_persons, cloud):

    #     skeletons_3d = []

    #     for detected_person in detected_persons:
    #         skeletons_3d.append(self.projectKeypoints3DCloud(detected_person, cloud))

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



