import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from Skeleton.utils_human_pose import KeyPointIndexTableMediapipe, Skeleton2d, Keypoint2D
import rospkg

from prediction_model import PredictionModel

class MediapipeModule():
    def __init__(self, model_name = "pose_landmarker_heavy", conf_thresh = 0.5):
        r = rospkg.RosPack()
        path = r.get_path('human_pose_detection')

        self.conf_thresh_ = conf_thresh
        num_poses, pres_conf, track_conf  = 4, 0.5, 0.5

        # num_poses : max number of poses that can be detected / integer [0, 5]  (default = 1)
        # min_pose_detection_confidence : minimum confidence score for the person detection / float [0.0, 1.0] (default = 0.5)
        # min_pose_presence_confidence : minimum confidence score of keypoint detection / float [0.0, 1.0] (default = 0.5)
        # min_tracking_confidence : minimum confidence score for the pose tracking / float [0.0, 1.0] (default = 0.5)

        base_options = python.BaseOptions(model_asset_path=path+'/' + model_name+ '.task')
        options = vision.PoseLandmarkerOptions(
            base_options=base_options, num_poses = num_poses, min_pose_detection_confidence = self.conf_thresh_, 
            min_pose_presence_confidence = pres_conf, min_tracking_confidence = track_conf, output_segmentation_masks=False)
        
        self.model = vision.PoseLandmarker.create_from_options(options)
        self.kp_table = KeyPointIndexTableMediapipe()
       

    def predictDetections(self, image_rgb, frame_id):

        skeletons_2d = []
        dim_rgb = image_rgb.shape

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)

        detections = self.model.detect(mp_image)

        #Normalized
        kp_n = detections.pose_landmarks

        # # World
        # skeletons_3d = []
        # kp_w = detections.pose_world_landmarks

        if(len(kp_n) != 0):

            # 2D normalized coordinates
            for i in range(0, len(kp_n)):
                detected_2d_skeleton = self.computeDetectionKeypoints(kp_n[i], dim_rgb, i, frame_id)
                skeletons_2d.append(detected_2d_skeleton)

            # 3D world coordinates computed by mediapipe
            # for i in range(0, len(kp_w)):
            #     detected_3d_skeleton = self.computeDetection3DWorldKeypoints(kp_w[i], i, frame_id, v_thresh)
            #     skeletons_3d.append(detected_3d_skeleton)

        return skeletons_2d
    
    def computeDetectionKeypoints(self, kp_n, dim_rgb, id, frame_id):

        skeleton_2d = Skeleton2d(id, frame_id)
        width, height, _ = dim_rgb

        for kp_index in range(0, len(kp_n)):
            
            label = self.kp_table.getKeypointName(kp_index)
            kp = kp_n[kp_index]
            #kp.z is computed but not used here as we have the depth image for a more precise value
            # if(kp.x != 0 and kp.y != 0 and kp.z != 0 and kp.visibility > visibility_thresh and kp.presence > presence_thresh):
            pix_x_rgb, pix_y_rgb  = int(height*kp.x), int(width*kp.y)
            
            if((pix_y_rgb < width ) & (pix_x_rgb < height)):
                new_kp = Keypoint2D(label, kp.visibility, pix_x_rgb, pix_y_rgb)
            else:
                new_kp = Keypoint2D(label, kp.visibility, 0, 0)

            skeleton_2d.addKeypoint(new_kp)
            
        return skeleton_2d
    
