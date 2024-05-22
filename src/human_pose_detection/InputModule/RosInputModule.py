import InputModule

import rospy
import cv2

import message_filters
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image, PointCloud2


class ROSInputModule(InputModule):
    def __init__(self, camera, mode, model):
        super.__init__(self, camera, model)
        self.bridge = CvBridge()
        self.mode_ = mode

        #Subscribers
        self.sub_rgb = message_filters.Subscriber("/camera/color/image_raw", Image) # frame : camera_color_optical_frame
        self.sub_depth = message_filters.Subscriber("/camera/depth/image_rect_raw", Image) # frame : camera_depth_optical_frame
        self.sub_aligned_depth = message_filters.Subscriber("/camera/aligned_depth_to_color/image_raw", Image) #frame : camera_color_optical_frame
        self.sub_confidence = message_filters.Subscriber("/camera/confidence/image_rect_raw", Image) # frame : camera_confidence_optical_frame
        self.sub_pointcloud = message_filters.Subscriber("/camera/depth_registered/points", PointCloud2) # frame : camera_color_optical_frame

        self.running_ = False

    # Changes the mode from pointcloud to depth image to get the z coordinates
    def run(self):
        self.running_ = True

        if(self.mode_ == "depth"):
            self.readRGBDepthFrames()
        elif(self.mode_ == "cloud"):
            self.readSyncRGBCloudFrames()
        else:
            rospy.loginfo("mode not supported")

    # Callbacks
    def readRGBDepthFrames(self, save_skeleton = False):
        # If rosbag recorded without power cord plugged in, latency between rgb and depth topics -> requires slop=1
        self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_rgb, self.sub_aligned_depth], queue_size=1, slop=0.05)
        self.ts.registerCallback(self.syncRGBDepthCallback)

    def syncRGBDepthCallback(self, image_rgb, image_depth):

        frame_id = image_rgb.header.frame_id

        frame_rgb = self.bridge.imgmsg_to_cv2(image_rgb, desired_encoding="bgr8")
        frame_depth = self.bridge.imgmsg_to_cv2(image_depth, desired_encoding="16UC1")


        # ============================= Depth post processing via Realsense API ================================
        # color_map = rs2.colorizer()
        # rs_frame = rs2.frame.
        # color_map.set_option(RS2_OPTION_HISTOGRAM_EQUALIZATION_ENABLED, 1.f)
        # color_map.set_option(RS2_OPTION_COLOR_SCHEME, 2.f)

        detected_persons_2d = self.model.predictDetections(frame_rgb, frame_id, 5, True)
        detected_persons_3d = self.model.projectDetectionsDepth(detected_persons_2d, frame_depth, True)

        self.user_callback(detected_persons_2d)

        self.cpt_ += 1

    # ================ Cloud version ---- Unused
    # def readSyncRGBCloudFrames(self):
    #     self.ts = message_filters.ApproximateTimeSynchronizer([self.sub_rgb, self.sub_pointcloud], queue_size=1, slop=0.1)
    #     self.ts.registerCallback(self.syncRGBCloudCallback)
    #     rospy.spin()
    
    # def syncRGBCloudCallback(self, image_rgb, cloud):

    #     frame_id = image_rgb.header.frame_id

    #     frame_rgb = self.bridge.imgmsg_to_cv2(image_rgb)

    #     detected_persons_2d = self.predictDetections(frame_rgb, frame_id, 5, True)
        
    #     detected_persons_3d = self.projectDetectionsCloud(detected_persons_2d, cloud, True)

   
   
