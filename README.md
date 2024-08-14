# human_pose_estimation

## Install package

1. git clone branch dev  
2. pip install requirements.txt
3. (optional) if you have the models on your machine, copy them into the models folder, else they will be downloaded via ultralytics upon launch

## Install Ultralytics on ROS-Noetic machines (Not Jetson)
1. pip install ultralytics

## Install Ultralytics on Jetson ORIN with Python 3.8 (JetPack 5.x)

1.  sudo apt update  
    sudo apt install python3-pip -y  
    pip install -U pip  
2.  pip install ultralytics[export]
3.  sudo reboot  

1.  pip uninstall torch torchvision
2.  sudo apt-get install -y libopenblas-base libopenmpi-dev  
    wget https://developer.download.nvidia.com/compute/redist/jp/v512/pytorch/torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl -O torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl  
    pip install torch-2.1.0a0+41361538.nv23.06-cp38-cp38-linux_aarch64.whl  
3.  sudo apt install -y libjpeg-dev zlib1g-dev  
    git clone https://github.com/pytorch/vision torchvision  
    cd torchvision  
    git checkout v0.16.2  
    python3 setup.py install --user  

1.  wget https://nvidia.box.com/shared/static/ zostg6agm00fb6t5uisw51qi6kpcuwzd.whl -O onnxruntime_gpu-1.17.0-cp38-cp38-linux_aarch64.whl  
    pip install onnxruntime_gpu-1.17.0-cp38-cp38-linux_aarch64.whl  

1. onnxruntime-gpu will automatically revert back the numpy version to latest. So we need to reinstall numpy to 1.23.5 to fix an issue by executing: pip install numpy==1.23.5  

## Launch

1. roslaunch human_pose_estimation rs_d435i.launch
2. roslaunch human_pose_estimation human_pose_estimation.launch

# Changing the model

In yolo_ros.py or yolo_cv.py, you can replace the model with one you would like between (yolo8x-pose.pt and yolox-pose-p6.pt).  
If you wish to use the .engine models (yolo8x-pose.engine and yolox-pose-p6.engine), you need to ensure that the Jetson specific installation process has been applied.