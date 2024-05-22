import matplotlib.pyplot as plt
import numpy as np

from sklearn.cluster import KMeans, MiniBatchKMeans, AffinityPropagation, MeanShift

import cv2

def compute_limb_sizes(skeleton, table):
    
    distances_limb = np.zeros(shape=(len(table.joint_table),1))

    for i in range(0, len(table.joint_table)):
        kp_pair = table.joint_table[i]
        kp_1 = skeleton.getKeypointByName(kp_pair[0])
        kp_2 = skeleton.getKeypointByName(kp_pair[1])

        if(kp_1.is_null_ == False and kp_2.is_null_ == False):
            d = np.sqrt((kp_2.x_ - kp_1.x_)**2 + (kp_2.y_ - kp_1.y_)**2 + (kp_2.z_ - kp_1.z_)**2)
        else:
            d = np.nan
        distances_limb[i]=d

    #print("distance table = ", distances_limb)
    return distances_limb

def compute_limb(skeletons, table):
    table_distances = []

    for i in range(0, len(skeletons)):
        # if(len(skeleton) == 1):
        #     dist = compute_limb_sizes(skeleton)
        # else:
        dist = compute_limb_sizes(skeletons[i][0], table)
        table_distances.append(dist)

   
    return table_distances

def compute_stat(skeletons, table, visualize = False):

    res_dist = compute_limb(skeletons, table)

    res_mean = np.nanmean(res_dist, axis=0)
    res_std = np.nanstd(res_dist, axis=0)
    res_var = np.nanvar(res_dist, axis=0)
    res_max = np.nanmax(res_dist, axis=0)
    res_min = np.nanmin(res_dist, axis=0)

    for i in range(0, len(table.joint_table)):
        print("Joint : ", table.joint_table[i], "| min : ", "%0.5f" % float(res_min[i]) ,"| max : ", "%0.5f" % float(res_max[i]),
               "| mean :" , "%0.5f" % float(res_mean[i]), "| std  :", "%0.5f" % float(res_std[i]), "| var : ", "%0.5f" % float(res_var[i]))
        
    if(visualize == True):
        x = np.linspace( start = 0, stop = len(res_dist), num = len(res_dist))
        y= []
        for joint in range(len(table.joint_table)):
            for i in range(0, len(res_dist)):
                y.append(res_dist[i][joint])
            plt.plot(x, y, label= table.joint_table[joint])
            y.clear()

        plt.axis([0, len(res_dist), 0, 0.6]) # [xmin, xmax, ymin, ymax]
        plt.legend(loc='upper right')
        plt.show()

    return [res_min, res_max, res_mean, res_std, res_var]

def compute_histogram(img_depth, save = False, path = None, filename = None):

    # filter out the zero values
    image_depth_list = img_depth[img_depth > 0]

    hist = cv2.calcHist([image_depth_list], [0], None, [65536], [0, 65536])

    fig = plt.figure()
    plt.plot(hist)
    plt.title("Grayscale Histogram")
    plt.xlabel("Bins")
    plt.ylabel("# of Pixels")
    plt.xlim([0, 6000])

    name = path + "/desk_standing2_hist/" + filename + "_hist.png"
    fig.savefig(name)

def compute_clusters(skeletons, img_depth):

    for i in range(0, len(skeletons)):
        print("detection ")
        for j in range(0, len(skeletons.keypoints)):
            kp_kmeans = KMeans(n_clusters=2, random_state=0, n_init="auto").fit(X)

