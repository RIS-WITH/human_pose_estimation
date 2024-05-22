import rospy

import numpy as np

import cv2

from visualization_msgs.msg import Marker, MarkerArray
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from std_msgs.msg import ColorRGBA
from geometry_msgs.msg import Point, Quaternion, Vector3

class RosVisualizationModule():
    def __init__(self, kp_table):
        self.pub_2dkp = rospy.Publisher('/rgb_w_keypoints', Image, queue_size=1)
        self.pub_3dkp = rospy.Publisher('/3d_keypoints', MarkerArray, queue_size=1)
        self.pub_2d_skeletons = rospy.Publisher('/rgb_w_skeletons', Image, queue_size=1)
        self.pub_3d_skeletons = rospy.Publisher('/3d_skeletons', MarkerArray, queue_size=1)

        self.bridge = CvBridge()
        self.kp_table_ = kp_table
    
    def publishKeypoint2D(self, image_rgb, skeletons, step = 2):
        frame_rgb = image_rgb.copy()
        dim_rgb = image_rgb.shape

        for skeleton in skeletons:
            #r,g,b = np.random.randint(0,255), np.random.randint(0,255), np.random.randint(0,255)
            r,g,b = (0,255,0)

            for keypoint in skeleton.keypoints:
                if(keypoint.is_null_ == True):
                    continue
                else:
                    if(keypoint.y_ + step > dim_rgb[0]) and (keypoint.x_ + step > dim_rgb[1]):
                        frame_rgb[keypoint.y_- step : -1, keypoint.x_ - step : -1, :] = (r,g,b)
                    elif(keypoint.y_ + step < dim_rgb[0]) and (keypoint.x_ + step > dim_rgb[1]):
                        frame_rgb[keypoint.y_- step : keypoint.y_+step, keypoint.x_ - step:-1, :] =  (r,g,b)
                    elif(keypoint.y_ + step > dim_rgb[0]) and (keypoint.x_ + step < dim_rgb[1]):
                        frame_rgb[keypoint.y_- step : -1, keypoint.x_ - step:keypoint.x_ + step, :] =  (r,g,b)
                    else:  
                        frame_rgb[keypoint.y_- step : keypoint.y_+step, keypoint.x_ - step:keypoint.x_ + step, :] =  (r,g,b)

        frame_rgb = self.bridge.cv2_to_imgmsg(frame_rgb)
        self.pub_2dkp.publish(frame_rgb)

    def publishKeypoint3D(self, skeletons3d):

        marker_array = MarkerArray()

        for skeleton in skeletons3d:

            #r,g,b = 1,1,1
            r,g,b = np.random.random(), np.random.random(), np.random.random()

            marker = Marker()
            marker.header.frame_id = str(skeleton.frame_id_) #"camera_color_optical_frame"
            marker.id = int(skeleton.skeleton_id_)

            marker.type = Marker.POINTS
            marker.action = Marker.MODIFY

            marker.color = ColorRGBA(r, g, b, 1.0)
            marker.scale = Vector3(0.02, 0.02, 0.02)
            marker.pose.position = Point(0,0,0)
            marker.pose.orientation = Quaternion(0,0,0,1)

            for i in range(0, len(skeleton.keypoints)):
                kp = skeleton.keypoints[i]
                if(kp.x_ == 0 and kp.y_ == 0 and kp.z_ == 0):
                    continue
                else:
                    point = Point(kp.x_, kp.y_, kp.z_)
                    marker.points.append(point)

            marker_array.markers.append(marker)

        self.pub_3dkp.publish(marker_array)

    def publishSkeleton2D(self, image_rgb, skeletons2d):
        # idea : have a different color for each detected person
        frame_rgb = image_rgb.copy()

        color = (0, 255, 0) 
        thickness = 3

        for skeleton in skeletons2d:
            for kp_pair in self.kp_table_.joint_table:
            
                kp_1 = skeleton.getKeypointByName(kp_pair[0])
                kp_2 = skeleton.getKeypointByName(kp_pair[1])
 
                if(kp_1.is_null_ == False and kp_2.is_null_ == False):
                    frame_rgb = cv2.line(frame_rgb, [kp_1.x_, kp_1.y_], [kp_2.x_, kp_2.y_], color, thickness)

        frame_rgb = self.bridge.cv2_to_imgmsg(frame_rgb)
        self.pub_2d_skeletons.publish(frame_rgb)

    def publishSkeleton3D(self, skeletons3d):
        # idea : have a different color for each detected person
        marker_array = MarkerArray()
       
        for skeleton in skeletons3d:
            
            # r,g,b = 1,1,1
            r,g,b = np.random.random(), np.random.random(), np.random.random()

            marker = Marker()
            marker.header.frame_id = str(skeleton.frame_id_) #"camera_color_optical_frame"
            marker.id = int(skeleton.skeleton_id_)

            marker.type = Marker.LINE_LIST
            marker.action = Marker.MODIFY

            marker.color = ColorRGBA(r, g, b, 1.0)
            marker.scale = Vector3(0.01, 0.01, 0.01)
            marker.pose.position = Point(0,0,0)
            marker.pose.orientation = Quaternion(0,0,0,1)

            for kp_pair in self.kp_table_.joint_table:
            
                kp_1 = skeleton.getKeypointByName(kp_pair[0])
                kp_2 = skeleton.getKeypointByName(kp_pair[1])

                if(kp_1.is_null_ == False and kp_2.is_null_ == False):

                    marker.points.append(Point(kp_1.x_, kp_1.y_, kp_1.z_))
                    marker.points.append(Point(kp_2.x_, kp_2.y_, kp_2.z_))

            marker_array.markers.append(marker)

        self.pub_3d_skeletons.publish(marker_array)
