import os
import rospkg
import numpy as np

import pyrealsense2 as rs2
import math
import struct
from sklearn.cluster import KMeans, MiniBatchKMeans, AffinityPropagation, MeanShift

import matplotlib.pyplot as plt

from human_pose_estimation.Skeleton.utils_human_pose import Keypoint3D, Skeleton3d, Skeleton3dClusterized, KeypointCandidates


class DetectionModule():
    def __init__(self, camera, model, masking_pattern = None, clustering_method = None):
        self.camera_ = camera
        self.model_ = model
        self.package_path = rospkg.RosPack().get_path('human_pose_estimation')

        #For saving purposes
        self.skeletons_array_ = []
        self.cpt_ = 0

        if(masking_pattern != None):
            self.mask_pattern_ = masking_pattern
        else:
            self.mask_pattern_ = ['square', 0]

        if(clustering_method != None):
            self.clustering_method_ = clustering_method
        else:
            self.clustering_method_ = "knn"

    def save_skeletons(self, folder_name, filename):
        # r = rospkg.RosPack()
        # package_path = r.get_path('human_pose_estimation')

        if(os.path.isdir(self.package_path + "/" + folder_name + "/") == False):
            creat_path = os.path.join(self.package_path, folder_name) 
            os.mkdir(creat_path) 

        name = self.package_path + "/" + folder_name + "/" + filename
        np.save(name + '.npy', np.array(self.skeletons_array_, dtype=object))
    
    def detect(self, frame_id, frame_rgb):

        detections = self.model_.predictDetections(frame_rgb, frame_id)
        
        return np.array(detections)
    
    def project(self, skeletons2d, frame_depth, take_min =False):

        projections, projections_cluster = self.projectDetectionsDepth(skeletons2d, frame_depth, take_min)
        print(projections.shape)
        return projections, projections_cluster
    
    def projectDetectionsDepth(self, detected_persons, img_depth, take_min = False):        
        skeletons_3d = np.empty(shape=(len(detected_persons)), dtype=Skeleton3d)
        skeletons_3d_cluster = np.empty(shape=(len(detected_persons)), dtype=Skeleton3dClusterized)

        # for detected_person in detected_persons:
        for index_detection in range(0, len(detected_persons)):
            skeleton3d, skeleton_3d_cluster = self.projectKeypoints3DDepth(detected_persons[index_detection], img_depth, take_min)
            skeletons_3d[index_detection] = skeleton3d
            skeletons_3d_cluster[index_detection] = skeleton_3d_cluster
        
        return skeletons_3d, skeletons_3d_cluster
    
    # Project the 2d keypoints in 3d keypoints
    def projectKeypoints3DDepth(self, detected_person, img_depth, take_min):

        skeleton_3d = Skeleton3d(detected_person.skeleton_id_, detected_person.frame_id_)
        skeleton_3d_cluster = Skeleton3dClusterized(detected_person.skeleton_id_, detected_person.frame_id_)

        # Loop over keypoints of the detected 2D skeleton
        for keypoint in detected_person.keypoints:
        
            # If both x and y values of the keypoint are null => return a kp with (0,0,0) as cooordinates
            if(keypoint.x_ == 0 and keypoint.y_ == 0):
                new_kp = Keypoint3D(keypoint.label_, keypoint.confidence_, 0, 0, 0)
            else:
                pix_x, pix_y = int(keypoint.x_), int(keypoint.y_)
                # images in numpy are HxWxC
                depth_value = img_depth[pix_y, pix_x]*self.camera_.depth_scale
                depth_point = rs2.rs2_deproject_pixel_to_point(self.camera_.color_intrinsic, [pix_x, pix_y], depth_value)

                # Max 3 clusters : (front, middle, background)
     
                new_kp = Keypoint3D(keypoint.label_, keypoint.confidence_, depth_point[0], depth_point[1], depth_point[2])
                
                # ================= Masking Step =================
                # get the value of the mask around the keypoint (x, y, depth_value)
                masked_values = self.extract_shape(image = img_depth, center = [pix_x, pix_y])

                # # project the values around the keypoint in 3D
                projected_values = self.project_shape(masked_values)

                # new_kp = self.applyMaskingPattern(img_depth, new_kp, pix_x, pix_y, take_min)

                if(depth_point[2] == 0):
                    # maximum distance of the depth camera parameters
                    current_min = 9.0
                else:
                    current_min = depth_point[2]
                # Loop over values and check for the min projected z value
                for projected_value in projected_values:
                    if(take_min and 0 < projected_value[2] < current_min):
                        new_x, new_y, new_z = projected_value[0], projected_value[1], projected_value[2]
                        new_kp = Keypoint3D(keypoint.label_, keypoint.confidence_, new_x, new_y, new_z)
                        current_min = new_z

                # if(keypoint.label_ == "right_ear"):
                #     print("pix init : ", pix_x, pix_y, "init depth = ", depth_value)
                #     print(masked_values)
                #     print(projected_values)
                #     print("old_kp before taking min : ", old_kp)
                #     print("new_kp after taking min : ", new_kp)


                # ================ Clustering Step ==================
                # projection inverse des clusters sur l'image depth
                # coloration de chaque cluster avec une couleur pour chacun
                # avec tous les keypoints

                # print(len(projected_values))
                # Here > 3 because we chose to have 3 clusters


                # if(len(projected_values) > 3 ):
                #     projected_z = []
                #     for value in projected_values:
                #         projected_z.append(value[2])

                #     clustering_model = KMeans(n_clusters= 3,  random_state=0, n_init=10)

                #     data = projected_values
                #     clustering_model.fit(data)
                #     #print("3d : ", clustering_model.cluster_centers_)

                #     np_projected = np.array(projected_z)
                #     np_projected = np_projected.reshape(-1, 1)
                #     data = np_projected
                #     clustering_model.fit(data)




                    #print("1d : ", clustering_model.cluster_centers_)

                    # print(keypoint.label_, keypoint.confidence_, keypoint.x_, keypoint.y_, keypoint.z_)
                    # keyp_cand = KeypointCandidates(keypoint.label_, keypoint.confidence_, depth_point[0], depth_point[1], depth_point[2])

                    # for point in clustering_model.cluster_centers_:
                    #     kp_cluster = Keypoint3D(keypoint.label_, keypoint.confidence_, point[0], point[1], point[2])
                    #     keyp_cand.candidates_.append(kp_cluster)
                        
                    # skeleton_3d_cluster.addKeypoint(keyp_cand)
            skeleton_3d.addKeypoint(new_kp)
                
        return skeleton_3d, skeleton_3d_cluster
    
    def applyMaskingPattern(self, image_depth, keypoint, pix_x, pix_y, take_min):
        new_kp = keypoint
        # get the value of the mask around the keypoint (x, y, depth_value)
        masked_values = self.extract_shape(image = image_depth, center = [pix_x, pix_y])

        # project the values around the keypoint in 3D
        projected_values = self.project_shape(masked_values)

        # print(projected_values)
        res = self.applyingClusteringMethod(projected_values)
        # print(res)

        if(keypoint.z_ == 0):
            # maximum distance of the depth camera parameters
            current_min = 9.0
        else:
            current_min = keypoint.z_
        # if(keypoint.label_ == "right_ear"):
        #     print("pix init : ", pix_x, pix_y, "init depth = ", depth_value)
        #     print(" initial value : ", depth_point)
        # Loop over values and check for the min projected z value
        for projected_value in projected_values:
            # if(keypoint.label_ == "right_ear"):
            #     print("for projected_value : ", projected_value)
            #     print("current_min: ",  current_min)
            if(take_min and 0 < projected_value[2] < current_min):
                # if(keypoint.label_ == "right_ear"):
                #     print("modified with ", projected_value[2])
                new_x, new_y, new_z = projected_value[0], projected_value[1], projected_value[2]
                new_kp = Keypoint3D(keypoint.label_, keypoint.confidence_, new_x, new_y, new_z)
                current_min = new_z

        return new_kp
    
    def applyingClusteringMethod(self, projected_keypoints):
        print("clustering method")

        clustering_model = KMeans(n_clusters= 3,  random_state=0, n_init=10)

        data = projected_keypoints
        # print(data.shape)
        clustering_model.fit(data)
        print(clustering_model.cluster_centers_)
        return clustering_model.cluster_centers_

    def reshapeSkeletonArray(self, skeletons_array):
        # print("len skeletons : ", len(skeletons) )
        # print("shape skeletons :", skeletons.shape)
        nb_timestep = len(skeletons_array)
        nb_skeleton = 1
        nb_keypoint = 17 # len(self.model.kp_table_.kp_table)
        table_keypoints = np.empty(shape=(nb_keypoint, nb_timestep), dtype=Keypoint3D)
        table_detections = np.empty(shape = nb_skeleton)


        table_keypoints[keypoint_index][cpt_timestep] = detected_skeletons[skeleton_index].keypoints[keypoint_index]

        
        cpt_timestep = 0
        for detected_skeletons in skeletons_array:
            for skeleton_index in range(0, len(detected_skeletons)):
                for keypoint_index in range(0, nb_keypoint):
                    table_keypoints[skeleton_index][keypoint_index][cpt_timestep] = detected_skeletons[skeleton_index].keypoints[keypoint_index]
            cpt_timestep += 1

        return table_keypoints

#========================== Masking methods ============================
    def extract_square_values_no_mask(self, image, center, length):
        x_center, y_center = center 
        values = []

        for x in range(x_center - length, x_center + length + 1):
            for y in range(y_center - length, y_center + length + 1):
                if((0 <= x < image.shape[1]) and (0 <= y < image.shape[0])):
                    depth_value = image[y, x]
                    pix_elem = [x, y, depth_value]
                    values.append(pix_elem)

        return np.array(values)
    
    def extract_circle_values_no_mask(self, image, center, radius):
        x_center, y_center = center
        values = []

        for x in range(x_center - radius, x_center + radius + 1):
            for y in range(y_center - radius, y_center + radius + 1):
                if((0 <= x < image.shape[1]) and (0 <= y < image.shape[0])):
                    if((x - x_center) ** 2 + (y - y_center) ** 2 <= radius ** 2):
                        depth_value = image[y, x]
                        pix_elem = [x, y, depth_value]
                        values.append(pix_elem)

        return np.array(values)

    def extract_shape(self, image, center = [0,0]):
        
        values = []
        if(self.mask_pattern_[0] == 'square'):
            values = self.extract_square_values_no_mask(image, center, self.mask_pattern_[1])
        elif (self.mask_pattern_[0] == 'circle'):
            values = self.extract_circle_values_no_mask(image, center, self.mask_pattern_[1])

        return values
    
    def project_shape(self, masked_values):
        projected_values = []

        for value in masked_values:
            depth_value = value[2]*self.camera_.depth_scale
            depth_point = rs2.rs2_deproject_pixel_to_point(self.camera_.color_intrinsic, [value[0], value[1]], depth_value)
            # Filter max and min distance of depth provided by Realsense Specs
            if(depth_point[2] > 0.25 and depth_point[2] < 9.0 ):
                projected_values.append(depth_point)
    
        return np.array(projected_values)
    

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