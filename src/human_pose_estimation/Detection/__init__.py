import numpy as np
import pyrealsense2 as rs2

# class Mask():
#     def __init__(self, center_x, center_y):
#         self.center_x_ = center_x
#         self.center_y_ = center_y
class PixelElement():
    def __init__(self, x, y, depth_value):
        self.x_ = x
        self.y_ = y
        self.depth_value_ = depth_value
    
    
class SquareMask():
    def __init__(self, length):
        self.length_ = length

    def extractValues(self, image, x_center, y_center):
        values = []

        for y in range(y_center - int(self.length_/2), y_center + int(self.length_/2) + 1):
            for x in range(x_center - int(self.length_/2), x_center + int(self.length_/2) + 1):
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    #new_pix = PixelElement(x, y, image[y, x])
                    depth_value = image[y, x]*0.001
                    depth_point = rs2.rs2_deproject_pixel_to_point(self.camera_.color_intrinsic, [x, y], depth_value)
                    if(depth_point[2] > 0.25 and depth_point[2] < 9.0 ):
                        values.append(depth_point)
                #values.append(image[y, x])

        return np.array(values)
    

class CircleMask():
    def __init__(self, radius):
        self.radius_ = radius

    def extractValues(self, image, x_center, y_center):
        values = []

        for y in range(y_center - self.radius_, y_center +  self.radius_ + 1):
            for x in range(x_center -  self.radius_, x_center +  self.radius_ + 1):
                if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                    if (x - x_center) ** 2 + (y - y_center) ** 2 <=  self.radius_ ** 2:
                        values.append(image[y, x])

        return np.array(values)
