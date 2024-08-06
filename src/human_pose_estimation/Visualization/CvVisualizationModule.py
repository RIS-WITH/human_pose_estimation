
import cv2
import matplotlib.pyplot as plt
import numpy as np

from matplotlib.patches import Circle
from matplotlib.pyplot import axline

# cerlce en maskage, donc passer en argument de detectionModule un objet du type voulu
# passer une list de trucs à afficher pour la visualisation des métriques (quels bones à afficher, )
# en fonction du keypoint, différentes formes radius (ex : oeil = circle) / dépendant de la distance aussi (proche = gros radius, loin = petit radius)
# à tester, prendre après les clusters le oe length le plus long
# 

class CvVisualizationModule():
    def __init__(self, kp_table, n_col, n_lin):
        # self.num_plot2d = 211
        
        # self.fig2, self.ax2 = plt.subplots(self.num_plot2d)
        # self.fig2d_ = {'': [self.fig2, self.ax2]}
        #self.fig2d_, self.ax2d_ = plt.subplots(self.num_plot2d)

        # ======== static figures : one plt.figure for each plot
        # self.fig2d_ = plt.figure()
        # self.ax2d_ = {'': self.fig2d_.add_subplot(self.num_plot2d)}

        # user gives the number of cols, row in args
        # intialize num_plot given those args
        # trust the user on the right number of figures

        #self.num_plot = 211
        self.num_plot = n_col*100 + n_lin*10
        # self.fig3d_ = plt.figure()

        # self.ax3d_ = {'': self.fig3d_.add_subplot(self.num_plot, projection='3d', title = 'default')}
        # self.ax3d_[''].view_init(elev=-90, azim=-90, roll=0)
        self.fig3d_ = plt.figure()

        self.ax3d_ = {}
    
        self.kp_table_ = kp_table

    def publishKeypoint2D(self, image_rgb, skeletons2d, radius_step = 3, fig_name = ''):

        # if fig_name in self.ax2d_.keys():
        #     ax2d_ = self.ax2d_[fig_name]
        # else:
        #     self.num_plot2d = self.num_plot2d + 1
        #     self.ax2d_[fig_name] = self.fig2d_.add_subplot(self.num_plot2d)

        #     ax2d_ = self.ax2d_[fig_name]

        if fig_name in self.ax3d_.keys():
            ax3d_ = self.ax3d_[fig_name]
        else:
            self.num_plot = self.num_plot + 1
            self.ax3d_[fig_name] = self.fig3d_.add_subplot(self.num_plot)
            ax3d_ = self.ax3d_[fig_name]

        ax3d_.clear()
        ax3d_.imshow(image_rgb)

        for detect_i in range(0, len(skeletons2d)):
            # Change color between detections here
            for kp in (skeletons2d[detect_i].keypoints):
                if(kp.is_null_ != True):
                    #cv2.circle(image_rgb, (kp.x_,kp.y_), radius=step, color=(0, 255, 0), thickness=-1)
                    circ = Circle((kp.x_,kp.y_), radius_step)
                    ax3d_.add_patch(circ)
                    #values = extract_square_values_no_mask(image_rgb[:, :, 0], center = [kp.x_, kp.y_], length = 3)
                    
                    # values = extract_circle_values(image_rgb[:, :, 0], center = [kp.x_, kp.y_], radius = 3)
                    # print("values 1 : ", values)
                    # print("len  1: ", len(values))
                    # values = extract_circle_values_no_mask(image_rgb[:,:,0], center = [kp.x_, kp.y_], radius = 3)
                    # print("values 2 : ", values)
                    # print("len  2: ", len(values))
        self.fig3d_.show()
        plt.pause(0.01)
        #imgplot = plt.imshow(img)

        # cv2.imshow("Skeleton detection", final_frame) 

        # key = cv2.waitKey(10)#pauses for 10 mseconds before fetching next image
        # if key == 27:#if ESC is pressed, exit loop
        #     cv2.destroyAllWindows()
        #     return

    def publishKeypoint3D(self, skeletons3d, fig_name = ''):

        if fig_name in self.ax3d_.keys():
            ax3d_ = self.ax3d_[fig_name]
        else:
            self.num_plot = self.num_plot + 1
            self.ax3d_[fig_name] = self.fig3d_.add_subplot(self.num_plot, projection='3d')
            self.ax3d_[fig_name].view_init(elev=-90, azim=-90, roll=0)

            ax3d_ = self.ax3d_[fig_name]

        ax3d_.clear()
        ax3d_.set_xlabel('X')
        ax3d_.set_ylabel('Y')
        ax3d_.set_zlabel('Z')
        ax3d_.set_xlim3d(-1.5, +1.5)
        ax3d_.set_ylim3d(-1.5, +1.5)
        ax3d_.set_zlim3d(-1.5, +1.5)
        # ax3d_.set_xlim3d(-0.5, +0.5)
        # ax3d_.set_ylim3d(-0.5, +0.5)
        # ax3d_.set_zlim3d(-0.5, +0.5)

        for detect_i in range(0, len(skeletons3d)):
            # Change color here
            xdata, ydata, zdata = [], [], []
            for kp in (skeletons3d[detect_i].keypoints):
                if(kp.is_null_ == False):
                    xdata.append(kp.x_)
                    ydata.append(kp.y_)
                    zdata.append(kp.z_)
            ax3d_.scatter3D(xdata, ydata, zdata)

        self.fig3d_.show()
        plt.pause(0.01)

    def publishSkeleton2D(self, image_rgb, skeletons2d, fig_name = ''):
        
        if fig_name in self.ax3d_.keys():
            ax3d_ = self.ax3d_[fig_name]
        else:
            self.num_plot = self.num_plot + 1
            self.ax3d_[fig_name] = self.fig3d_.add_subplot(self.num_plot)
            ax3d_ = self.ax3d_[fig_name]
            self.ax3d_[fig_name].view_init(elev=-90, azim=-90, roll=0)

        ax3d_.clear()
        ax3d_.imshow(image_rgb)

        for detect_i in range(0, len(skeletons2d)):
            # Change color between detections here
            for kp_pair in self.kp_table_.joint_table:
                kp_1 = skeletons2d[detect_i].getKeypointByName(kp_pair[0])
                kp_2 = skeletons2d[detect_i].getKeypointByName(kp_pair[1])
    
                if(kp_1.is_null_ == False and kp_2.is_null_ == False):
                    # ax2d_.add_patch(circ)
                    # ax2d_.plot()
                    #line2d = axline((kp_1.x_, kp_1.y_), (kp_2.x_, kp_2.y_), linewidth=2, color='r')
                    point_x = [kp_1.x_, kp_2.x_]
                    point_y = [kp_1.y_, kp_2.y_]
                    # ax2d_.plot(line2d)

                    ax3d_.plot(point_x, point_y, color="green", linewidth=1)
                    #cv2.line(image_rgb, [kp_1.x_, kp_1.y_], [kp_2.x_, kp_2.y_], (0, 255, 0), thickness=1, lineType=1)

        self.fig3d_.show()
        plt.pause(0.01)
        # final_frame = image_rgb

        # cv2.imshow("Skeleton detection", final_frame) 

        # key = cv2.waitKey(10)#pauses for 10 mseconds before fetching next image
        # if key == 27:#if ESC is pressed, exit loop
        #     cv2.destroyAllWindows()
        #     return

    def publishSkeleton3D(self, skeletons3d, fig_name = ''):
        
        if fig_name in self.ax3d_.keys():
            ax3d_ = self.ax3d_[fig_name]
        else:
            self.num_plot = self.num_plot + 1
            self.ax3d_[fig_name] = self.fig3d_.add_subplot(self.num_plot, projection='3d')
            ax3d_ = self.ax3d_[fig_name]
            self.ax3d_[fig_name].view_init(elev=-90, azim=-90, roll=0)
    
        ax3d_.clear()
        ax3d_.set_xlabel('X')
        ax3d_.set_ylabel('Y')
        ax3d_.set_zlabel('Z')
        ax3d_.set_xlim3d(-1.5, +1.5)
        ax3d_.set_ylim3d(-1.5, +1.5) 
        ax3d_.set_zlim3d(-1.5, +1.5)
        # ax3d_.set_xlim3d(-0.5, +0.5)
        # ax3d_.set_ylim3d(-0.5, +0.5)
        # ax3d_.set_zlim3d(-0.5, +0.5)
        
        for detect_i in range(0, len(skeletons3d)):
            for kp_pair in self.kp_table_.joint_table:
                kp_1 = skeletons3d[detect_i].getKeypointByName(kp_pair[0])
                kp_2 = skeletons3d[detect_i].getKeypointByName(kp_pair[1])

                if(kp_1.is_null_ == False and kp_2.is_null_ == False):
                    ax3d_.plot([kp_1.x_, kp_2.x_], [kp_1.y_, kp_2.y_], zs = [kp_1.z_, kp_2.z_], color = 'b')
        self.fig3d_.show()
        plt.pause(0.01)

    # def displayStatistics(self, skeletons3d, fig_name = ''):
