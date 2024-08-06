import matplotlib.pyplot as plt
import numpy as np

import cv2

class EvaluationModule():
    def __init__(self, kp_table):
        self.kp_table_ = kp_table

    def computeLimb(self, skeletons):
        nb_joints = len(self.kp_table_.joint_table)
        nb_timestep = len(skeletons)
        nb_skeleton = 1
        table_distances = np.empty(shape=(nb_skeleton, nb_joints, nb_timestep))

        # Loop over timesteps
        for time_index in range(0, nb_timestep):
            # Loop over skeletons at each timestep
            for skeleton_index in range(0, nb_skeleton):
                # Loop over joints in the skeleton
                for joint_index in range(0, nb_joints):
                    kp_pair = self.kp_table_.joint_table[joint_index]
                    kp_1 = skeletons[time_index][skeleton_index].getKeypointByName(kp_pair[0])
                    kp_2 = skeletons[time_index][skeleton_index].getKeypointByName(kp_pair[1])

                    if(kp_1.is_null_ == False and kp_2.is_null_ == False):
                        d = np.sqrt((kp_2.x_ - kp_1.x_)**2 + (kp_2.y_ - kp_1.y_)**2 + (kp_2.z_ - kp_1.z_)**2)
                    else:
                        d = np.nan

                    table_distances[skeleton_index][joint_index][time_index] = d

        return table_distances

    def computeJointStatistics(self, skeletons, fig_name = ''):
        
        # res_dist contains [timestep, joints_distance]
        res_joint_distances = self.computeLimb(skeletons)

        nb_metrics = 6
        nb_joints = len(self.kp_table_.joint_table)
        nb_skeleton = 1
        metrics = np.empty(shape=(nb_skeleton, nb_joints, nb_metrics))
        res_joints_occurences = np.empty(shape=(nb_skeleton, nb_joints))
        nb_timestep = 0
        
        for skeleton in range(0, res_joint_distances.shape[0]):
            for joint in range(0, res_joint_distances.shape[1]):
                selected_elem = res_joint_distances[skeleton, joint, :]
                nb_timestep = len(selected_elem)
                metrics[skeleton, joint, 0] = np.nanmin(selected_elem)
                metrics[skeleton, joint, 1] = np.nanmax(selected_elem)
                metrics[skeleton, joint, 2] = np.nanmean(selected_elem)
                metrics[skeleton, joint, 3] = np.nanstd(selected_elem)
                metrics[skeleton, joint, 4] = np.nanvar(selected_elem)
                metrics[skeleton, joint, 5] = np.sum(~np.isnan(selected_elem), axis= 0)
                # Extract only the number of occurences
                res_joints_occurences[skeleton, joint] = (np.sum(~np.isnan(selected_elem), axis= 0)/nb_timestep)*100

        for skeleton in range(0, res_joint_distances.shape[0]):
            for i in range(0, len(self.kp_table_.joint_table)):
                print("Joint : ", self.kp_table_.joint_table[i], "| min : ", "%0.5f" % float(metrics[skeleton][i][0]) ,"| max : ",
                    "%0.5f" % float(metrics[skeleton][i][1]), "| mean :" , "%0.5f" % float(metrics[skeleton][i][2]),
                    "| std  :", "%0.5f" % float(metrics[skeleton][i][3]), "| var : ", "%0.5f" % float(metrics[skeleton][i][4]),
                    "| nb_detections : ", int(metrics[skeleton][i][5]), "/", nb_timestep)

        return res_joint_distances, res_joints_occurences
    
    def computeKeypointStatistics(self, skeletons):
        nb_keypoint = len(self.kp_table_.kp_table)
        nb_timestep = len(skeletons)
        nb_skeleton = 1
        # table_keypoints = np.empty(shape=(nb_skeleton, nb_keypoint, 4,  nb_timestep))
        table_keypoints = np.empty(shape=(nb_skeleton, nb_keypoint,  nb_timestep))
        res_keypoints_occurences = np.empty(shape=(nb_skeleton, nb_keypoint), dtype = bool)
        # print("skeletons", skeletons[0])

        for skeleton_index in range(0, nb_skeleton):
            for skeleton in skeletons[skeleton_index]:
                print(skeleton)
                for keypoint in range(0, len(skeleton)):
                    for timestep in range(0, nb_timestep):
                        print(skeleton[keypoint][timestep])
                        if(skeleton[keypoint][timestep].is_null_ == False):
                            table_keypoints[skeleton_index][keypoint][timestep] = True
                        else:
                            table_keypoints[skeleton_index][keypoint][timestep] = False

        # for timestep in range(0, nb_timestep):
            
        #     for detection in range(0, len(skeletons[timestep])):
        #         for keypoint in range(0, nb_keypoint):

        #             skele
        #             if(skeletons[timestep][detection].keypoints[keypoint].is_null_ == False):
        #                 table_keypoints[detection][keypoint][timestep] = True
        #             else:
        #                 table_keypoints[detection][keypoint][timestep] = False

        # print(table_keypoints)


        # for skeleton in range(0, nb_skeleton):
        #     for timestep in range(0, nb_timestep):
        #         for keypoint in range(0, nb_keypoint):
        #             if(skeletons[timestep][skeleton].keypoints[keypoint].is_null_ == False):
        #                 table_keypoints[skeleton][keypoint][timestep] = True
        #             else:
        #                 table_keypoints[skeleton][keypoint][timestep] = False

        for skeleton in range(0, nb_skeleton):
            for keypoint in range(0, nb_keypoint):
                selected_elem = table_keypoints[skeleton, keypoint, :]
                occurences = np.count_nonzero(selected_elem)
                ratio = (occurences/nb_timestep)*100
                res_keypoints_occurences[skeleton, keypoint] = ratio
                #print("Keypoint :", self.kp_table_.kp_table[keypoint], " Detection ratio : ", ratio, "%")
               
        # for skeleton in range(0, nb_skeleton):
        #     for keypoint in range(0, nb_keypoint):
        #         selected_elem = table_keypoints[skeleton, keypoint, :, :]
        #         print(selected_elem)
        #         occurences = np.sum(~np.isnan(selected_elem), axis = 1)
        #         ratio = occurences/nb_timestep
                #print("Keypoint :", self.kp_table_.kp_table[keypoint], " Detection ratio : ", ratio)
        return res_keypoints_occurences
    
    def compareKeypointDetections(self, init_detections, post_detections):
        print("====== Comparison between initial and post keypoint detections : =======")

        for skeleton in range(0, init_detections.shape[0]):
            print("\nFor skeleton ", skeleton, " :")
            for keypoint in range(0, init_detections.shape[1]):
                print("For kp : ", self.kp_table_.kp_table[keypoint], "before : ", init_detections[skeleton, keypoint] , "%", "after :", post_detections[skeleton, keypoint], "%" )

    def compareJointDetections(self, init_detections, post_detections):
        print("====== Comparison between initial and post joint detections : =======")

        for skeleton in range(0, init_detections.shape[0]):
            print("\nFor skeleton ", skeleton, " :")
            for joint in range(0, init_detections.shape[1]):
                print("For joint : ", self.kp_table_.joint_table[joint], "before : ", init_detections[skeleton, joint] , "%", "after :", post_detections[skeleton, joint], "%" )

    def displayLimbLength(self, limb_lengths, fig_name = '', save_plot = False, filename = ''):
        fig_res = plt.figure(figsize=(19,10))

        nb_timestep = len(limb_lengths[0])

        if(fig_name != ''):
            x = np.linspace(start = 0, stop = nb_timestep, num = nb_timestep)
            y = []
            
            for joint in range(len(self.kp_table_.joint_table)):
                y = limb_lengths[joint][:]
                plt.plot(x, y, label= self.kp_table_.joint_table[joint])

            plt.axis([0, nb_timestep, 0, 0.6]) # [xmin, xmax, ymin, ymax]
            fig_res.legend(loc='upper right')
            fig_res.show()
            plt.pause(0.5)

        if(save_plot == True):
            print("saving plot")
            plt.savefig(self.package_path + '/data/results/' + filename)

# Compute the percentage of improvement between non-detections and post-detections

# def compute_histogram(img_depth, save = False, path = None, filename = None):

#     # filter out the zero values
#     image_depth_list = img_depth[img_depth > 0]

#     hist = cv2.calcHist([image_depth_list], [0], None, [65536], [0, 65536])

#     fig = plt.figure()
#     plt.plot(hist)
#     plt.title("Grayscale Histogram")
#     plt.xlabel("Bins")
#     plt.ylabel("# of Pixels")
#     plt.xlim([0, 6000])

#     name = path + "/desk_standing2_hist/" + filename + "_hist.png"
#     fig.savefig(name)

# def compute_clusters(skeletons, img_depth):

#     for i in range(0, len(skeletons)):
#         print("detection ")
#         for j in range(0, len(skeletons.keypoints)):
#             kp_kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)

