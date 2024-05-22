from ultralytics import YOLO
import rospkg

from human_pose_detection.Skeleton.utils_human_pose import KeyPointIndexTableYolo, Skeleton2d, Keypoint2D 


class Yolov8Module():
    def __init__(self, model_name = "yolov8x-pose-p6", conf_thresh = 0.25 ):
        r = rospkg.RosPack()
        path = r.get_path('human_pose_detection')

        self.model = YOLO(model = path + "/models/yolo/" + model_name + ".pt", verbose=False)
        self.kp_table = KeyPointIndexTableYolo()
        self.conf_thresh_ = conf_thresh

    def predictDetections(self, image_rgb, frame_id):
        
        detections = self.model.predict(image_rgb, conf = self.conf_thresh_, show = False)
        dim_rgb = image_rgb.shape

        skeletons_2d = []

        # we have to take the first element for no reason
        kp_n = detections[0].keypoints.xyn
        conf_n = detections[0].keypoints.conf

        if((kp_n is not None) & (conf_n is not None)):
            kp_n = kp_n.cpu().data.numpy()
            conf_n = conf_n.cpu().data.numpy()
    
            for i in range(0, len(kp_n)):
                detected_2ds_skeleton = self.computeDetectionKeypoints(kp_n[i], conf_n[i], dim_rgb, i, frame_id)
                skeletons_2d.append(detected_2ds_skeleton)
    
        return skeletons_2d
    
    def computeDetectionKeypoints(self, kp_n, conf_n, dim_rgb, id, frame_id):

        skeleton_2d = Skeleton2d(id, frame_id)
        width, height, _ = dim_rgb

        for kp_index in range(0, len(kp_n)):
            label = self.kp_table.getKeypointName(kp_index)
            kp = kp_n[kp_index]

            pix_x_rgb, pix_y_rgb  = int(height*kp[0]), int(width*kp[1])
            #print("label : ", label, " conf : ", conf_n[kp_index], " normalized : ", kp[0], kp[1], "pixel values :", pix_x_rgb, pix_y_rgb)

            if((pix_y_rgb < width) & (pix_x_rgb < height)):
                new_kp = Keypoint2D(label, conf_n[kp_index], pix_x_rgb, pix_y_rgb)
            else:
                new_kp = Keypoint2D(label, conf_n[kp_index], 0, 0)

            skeleton_2d.addKeypoint(new_kp)

        return skeleton_2d
