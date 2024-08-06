
import cv2
import matplotlib.pyplot as plt
import numpy as np

def displaySkeletons2D(img_rgb, img_depth, skeletons, table):

    for detect_i in range(0, len(skeletons)):
        for kp in (skeletons[detect_i].keypoints):
            if(kp.is_null_ != True):
                cv2.circle(img_rgb, (kp.x_,kp.y_), radius=2, color=(0, 255, 0), thickness=-1)

        for kp_pair in table.joint_table:
            kp_1 = skeletons[detect_i].getKeypointByName(kp_pair[0])
            kp_2 = skeletons[detect_i].getKeypointByName(kp_pair[1])
 
            if(kp_1.is_null_ == False and kp_2.is_null_ == False):
                cv2.line(img_rgb, [kp_1.x_, kp_1.y_], [kp_2.x_, kp_2.y_], (0, 255, 0), thickness=1, lineType=1)

    #print(img_rgb.shape, img_depth.shape)

    img_depth_bgr = cv2.cvtColor(img_depth, cv2.COLOR_GRAY2BGR)

    img_depth_bgr = img_depth_bgr.astype(np.uint8)
    # ============= issue depth is uint16 and rgb is uint8 so cannot concat them together
    # img_rgb = img_rgb.astype(np.uint16)
    # img_depth_bgr = img_depth_bgr.astype(np.uint16)

    final_frame = cv2.vconcat((img_rgb, img_depth_bgr))
        
    smaller_img = final_frame[::2, ::2]
    #resized = cv2.resize(final_frame, (960,480))

    
    # Displaying the image 
    cv2.imshow("Skeleton detection", smaller_img) 

    #cv2.imshow('detected skeletons', img_rgb) 
    #cv2.waitKey(0)

    key = cv2.waitKey(10)#pauses for 10 mseconds before fetching next image
    if key == 27:#if ESC is pressed, exit loop
        cv2.destroyAllWindows()
        return

def displaySkeletons3D(skeletons, table, fig, ax):

    ax.clear()
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_xlim3d(-1.5, +1.5)
    ax.set_ylim3d(-1.5, +1.5) 
    ax.set_zlim3d(-1.5, +1.5)

    for detect_i in range(0, len(skeletons)):
        # ==============plot keypoints ==============
        xdata, ydata, zdata = [], [], []
        for kp in (skeletons[detect_i].keypoints):
            if(kp.is_null_ == False):
                xdata.append(kp.x_)
                ydata.append(kp.y_)
                zdata.append(kp.z_)
        ax.scatter3D(xdata, ydata, zdata)
    #     fig.show()
    # plt.pause(0.1)
        #============ plot skeletons ===============
        for kp_pair in table.joint_table:
            kp_1 = skeletons[detect_i].getKeypointByName(kp_pair[0])
            kp_2 = skeletons[detect_i].getKeypointByName(kp_pair[1])

            if(kp_1.is_null_ == False and kp_2.is_null_ == False):
                #print(kp_pair)
                ax.plot([kp_1.x_, kp_2.x_], [kp_1.y_, kp_2.y_], zs = [kp_1.z_, kp_2.z_], color = 'b')
                # joint = Line3D(xs = [kp_1.x_, kp_2.x_], ys = [kp_1.y_, kp_2.y_], zs = [kp_1.z_, kp_2.z_])
                # ax.plot(joint, color = 'b')
                # ax.plot(kp_1.x_ + kp_2.x_, kp_1.y_+ kp_2.y_, kp_1.z_+ kp_2.z_, color = 'b')
    fig.show()
    plt.pause(5)

# def displayClusters(skeletons, img_rgb, img_depth):
