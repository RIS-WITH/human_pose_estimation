import numpy as np
import rospy

class SkeletonsHandler:
    def __init__(self):
        self.skeletons = None
        self.handler_node =  rospy.init_node('skeletons_handler_node')

#================== KeypointIndexTable structures =================
 
class KeypointIndexTable:
    def getKeypointName(self, id):
        return self.kp_table[id]
    
    def getKeypointId(self, name):
        return self.kp_table.index(name)
    
class KeyPointIndexTableMediapipe(KeypointIndexTable):
    #passer la joint table en index : [0,4]
    def __init__(self):
        self.kp_table = ["nose", "left_eye_inner", "left_eye", "left_eye_outer", "right_eye_inner", "right_eye", "right_eye_outer", 
                        "left_ear", "right_ear", "mouth_left", "mouth_right", "left_shoulder", "right_shoulder", "left_elbow", 
                        "right_elbow", "left_wrist", "right_wrist", "left_pinky", "right_pinky", "left_index", "right_index", "left_thumb",
                        "right_thumb", "left_hip", "right_hip", "left_knee", "right_knee", "left_ankle", "right_ankle", "left_heel", 
                        "right_heel", "left_foot_index", "right_foot_index"]
        
        # maybe add ["left_shoulder","left_hip"], ["right_shoulder","right_hip"] but not proper joints
        self.joint_table = [["nose", "left_eye_inner"], ["left_eye_inner", "left_eye"], ["left_eye","left_eye_outer"], ["nose", "right_eye_inner"],
                            ["right_eye_inner", "right_eye"], ["right_eye", "right_eye_outer"], ["left_eye_outer", "left_ear"],
                            ["right_eye_outer", "right_ear"], ["mouth_left", "mouth_right"], ["left_shoulder", "right_shoulder"],
                            ["left_shoulder", "left_elbow"], ["right_shoulder", "right_elbow"],  ["left_elbow", "left_wrist"],
                            ["right_elbow", "right_wrist"], ["left_wrist", "left_pinky"], ["right_wrist", "right_pinky"],  
                            ["left_wrist", "left_index"], ["right_wrist", "right_index"], ["left_pinky", "left_index"],
                            ["right_pinky", "right_index"], ["left_wrist", "left_thumb"], ["right_wrist", "right_thumb"],
                            ["left_hip", "right_hip"], ["left_hip", "left_knee"], ["right_hip", "right_knee"], ["left_knee", "left_ankle"],
                            ["right_knee", "right_ankle"], ["left_ankle", "left_heel"], ["right_ankle", "right_heel"], ["left_ankle", "left_foot_index"],
                            ["right_ankle", "right_foot_index"], ["left_heel", "left_foot_index"], ["right_heel", "right_foot_index"]]
        
class KeyPointIndexTableYolo(KeypointIndexTable):
    def __init__(self):
        self.kp_table = ["nose", "left_eye", "right_eye", "left_ear", "right_ear", "left_shoulder", "right_shoulder", 
                        "left_elbow", "right_elbow", "left_wrist", "right_wrist", "left_hip", "right_hip", "left_knee", 
                        "right_knee", "left_ankle", "right_ankle"]
        
        # maybe add ["left_shoulder","left_hip"], ["right_shoulder","right_hip"] but not proper joints
        self.joint_table = [["nose", "left_eye"], ["nose", "right_eye"], ["left_eye","left_ear"], ["right_eye","right_ear"],
                            ["left_shoulder","right_shoulder"], ["left_shoulder","left_elbow"], ["right_shoulder","right_elbow"],
                            ["left_elbow","left_wrist"], ["right_elbow","right_wrist"], ["left_hip","right_hip"], ["left_hip","left_knee"],
                            ["right_hip","right_knee"], ["left_knee","left_ankle"], ["right_knee","right_ankle"]]

#================== Skeleton structures =================

# idea : create different types of skeletons classes given the keypoints model used

class Skeleton:
    def __init__(self, id, frame_id):
        self.skeleton_id_ = id
        self.keypoints = []
        self.frame_id_ = frame_id
    
    def getKeypointByIndex(self, index):
        return self.keypoints[index]

    def getKeypointByName(self, name):
        for kp in self.keypoints:
            if(kp.label_ == name):
                return kp
            
    def __str__(self):
        keyp_str =""
        for kp in self.keypoints:
            keyp_str+=kp.__str__() +" \n"
        return keyp_str

#keypoints coordinates in pixels values
class Skeleton2d(Skeleton):
    def __init__(self, id, frame_id):
        super().__init__(id, frame_id)

    def addKeypoint(self, keypoint):
        self.keypoints.append(keypoint)
      
#keypoints coordinates in (x,y,z) values
# idea => have multiple candidates keypoints for each joint

class Skeleton3d(Skeleton):
    def __init__(self, id, frame_id):
        super().__init__(id, frame_id)

    def addKeypoint(self, keypoint):
        self.keypoints.append(keypoint)

# ================== Keypoint structures ===================
# idea : print (with a plot) variance and std over each limb sizes to see how it varies
# idea : compute limb sizes

class JointSegment:
    def __init__(self, keypoint_1, keypoint_2):
        self.keypoint_1_ = keypoint_1
        self.keypoint_2_ = keypoint_2

# frame_id
class Keypoint:
    def __init__(self, label, conf):
        self.label_ = label
        self.confidence_ = conf
        self.is_null_ = False

class Keypoint2D(Keypoint):
    def __init__(self, label, conf, x, y):
        super().__init__(label, conf)
        self.x_ = x
        self.y_ = y
        if(x == 0 and y == 0):
            self.is_null_ = True
    
    def __str__(self):
        return "Keypoint2D: label: {0} conf: {1} (pix_x:{2},pix_y:{3}) is_null :{4} ".format(self.label_, self.confidence_, self.x_, self.y_, self.is_null_)

class Keypoint3D(Keypoint2D):
     def __init__(self, label, conf, x, y, z):
        super().__init__(label, conf, x, y)
        self.z_ = z
        if(x == 0 and y == 0 and z == 0):
            self.is_null_ = True

     def __str__(self):
        return "Keypoint3D: label: {0} conf: {1} (x:{2},y:{3},z:{4}) is_null :{5} ".format(self.label_, self.confidence_, self.x_, self.y_, self.z_, self.is_null_)
     
class KeypointCandidates(Keypoint):
    def __init__(self, label, conf, nb_neighboors):
        self.label_ = label
        self.confidence_ = conf
        self.is_null_ = False
        self.neighboors = np.empty(nb_neighboors, dtype = Keypoint3D)

# class SkeletonCandidates(Skeleton):
#     def __init__(self, id, frame_id):
#         self.skeleton_id_ = id
#         self.keypoints = []
#         self.frame_id_ = frame_id
    