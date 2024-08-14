import pyrealsense2 as rs
import numpy as np
import cv2

from ultralytics import YOLO

#Load the model
model_name = "yolo8x-pose.pt"
model = YOLO(model = "/models/yolo/" + model_name, verbose=False)

# Configure depth and color streams
pipeline = rs.pipeline()
config = rs.config()

# Get device product line for setting a supporting resolution
pipeline_wrapper = rs.pipeline_wrapper(pipeline)
pipeline_profile = config.resolve(pipeline_wrapper)
device = pipeline_profile.get_device()
device_product_line = str(device.get_info(rs.camera_info.product_line))

found_rgb = False
for s in device.sensors:
    if s.get_info(rs.camera_info.name) == 'RGB Camera':
        found_rgb = True
        break
if not found_rgb:
    print("The demo requires Depth camera with Color sensor")
    exit(0)

config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

# Start streaming
pipeline.start(config)

try:
    while True:

        frames = pipeline.wait_for_frames()
        color_frame = frames.get_color_frame()
        if not color_frame:
            continue

        # Convert image to numpy arrays
        color_image = np.asanyarray(color_frame.get_data())

        result = model.predict(color_image, show = True)

finally:

    # Stop streaming
    pipeline.stop()