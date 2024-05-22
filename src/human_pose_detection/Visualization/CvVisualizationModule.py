
import cv2
import matplotlib.pyplot as plt
import numpy as np

class CvVisualizationModule():
    def __init__(self, kp_table):
        # self.fig2d_ = plt.figure()
        # self.ax2d_ = plt.axes()

        self.fig3d_ = plt.figure()

        self.ax3d_ = self.fig3d_.add_subplot(projection='3d')
        self.ax3d_.view_init(elev=-90, azim=-90, roll=0)

        self.kp_table_ = kp_table

    def publishKeypoint2D(self, image_rgb, skeletons2d, step = 2):

        for detect_i in range(0, len(skeletons2d)):
            # Change color between detections here
            for kp in (skeletons2d[detect_i].keypoints):
                if(kp.is_null_ != True):
                    cv2.circle(image_rgb, (kp.x_,kp.y_), radius=step, color=(0, 255, 0), thickness=-1)

        #final_frame = image_rgb[::2]
        final_frame = image_rgb

        cv2.imshow("Skeleton detection", final_frame) 

        key = cv2.waitKey(10)#pauses for 10 mseconds before fetching next image
        if key == 27:#if ESC is pressed, exit loop
            cv2.destroyAllWindows()
            return

    def publishKeypoint3D(self, skeletons3d):

        self.ax3d_.clear()
        self.ax3d_.set_xlabel('X')
        self.ax3d_.set_ylabel('Y')
        self.ax3d_.set_zlabel('Z')
        self.ax3d_.set_xlim3d(-1.5, +1.5)
        self.ax3d_.set_ylim3d(-1.5, +1.5)
        self.ax3d_.set_zlim3d(-1.5, +1.5)

        for detect_i in range(0, len(skeletons3d)):
            # Change color here
            xdata, ydata, zdata = [], [], []
            for kp in (skeletons3d[detect_i].keypoints):
                if(kp.is_null_ == False):
                    xdata.append(kp.x_)
                    ydata.append(kp.y_)
                    zdata.append(kp.z_)
            self.ax3d_.scatter3D(xdata, ydata, zdata)

        self.fig3d_.show()
        plt.pause(0.5)

    def publishSkeleton2D(self, image_rgb, skeletons2d):

        for detect_i in range(0, len(skeletons2d)):
            # Change color between detections here
            for kp_pair in self.kp_table_.joint_table:
                kp_1 = skeletons2d[detect_i].getKeypointByName(kp_pair[0])
                kp_2 = skeletons2d[detect_i].getKeypointByName(kp_pair[1])
    
                if(kp_1.is_null_ == False and kp_2.is_null_ == False):
                    cv2.line(image_rgb, [kp_1.x_, kp_1.y_], [kp_2.x_, kp_2.y_], (0, 255, 0), thickness=1, lineType=1)

        final_frame = image_rgb

        cv2.imshow("Skeleton detection", final_frame) 

        key = cv2.waitKey(10)#pauses for 10 mseconds before fetching next image
        if key == 27:#if ESC is pressed, exit loop
            cv2.destroyAllWindows()
            return

    def publishSkeleton3D(self, skeletons3d):

        self.ax3d_.clear()
        self.ax3d_.set_xlabel('X')
        self.ax3d_.set_ylabel('Y')
        self.ax3d_.set_zlabel('Z')
        self.ax3d_.set_xlim3d(-1.5, +1.5)
        self.ax3d_.set_ylim3d(-1.5, +1.5) 
        self.ax3d_.set_zlim3d(-1.5, +1.5)
        
        for detect_i in range(0, len(skeletons3d)):
            for kp_pair in self.kp_table_.joint_table:
                kp_1 = skeletons3d[detect_i].getKeypointByName(kp_pair[0])
                kp_2 = skeletons3d[detect_i].getKeypointByName(kp_pair[1])

                if(kp_1.is_null_ == False and kp_2.is_null_ == False):
                    self.ax3d_.plot([kp_1.x_, kp_2.x_], [kp_1.y_, kp_2.y_], zs = [kp_1.z_, kp_2.z_], color = 'b')
        self.fig3d_.show()
        plt.pause(0.5)

