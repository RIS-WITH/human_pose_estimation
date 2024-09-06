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

1.  wget https://nvidia.box.com/shared/static/zostg6agm00fb6t5uisw51qi6kpcuwzd.whl -O onnxruntime_gpu-1.17.0-cp38-cp38-linux_aarch64.whl  
    pip install onnxruntime_gpu-1.17.0-cp38-cp38-linux_aarch64.whl  

1. onnxruntime-gpu will automatically revert back the numpy version to latest. So we need to reinstall numpy to 1.23.5 to fix an issue by executing: pip install numpy==1.23.5  

## Launch

1. roslaunch human_pose_estimation rs_d435i.launch
2. roslaunch human_pose_estimation human_pose_estimation.launch

# Changing the model

In yolo_ros.py or yolo_cv.py, you can replace the model with one you would like between (yolo8x-pose.pt and yolox-pose-p6.pt).  
If you wish to use the .engine models (yolo8x-pose.engine and yolox-pose-p6.engine), you need to ensure that the Jetson specific installation process has been applied.

## Installation StereoLabs ZEDX om Jetson Orin

# Setup GMSL2 driver (https://www.stereolabs.com/docs/get-started-with-zed-link/install-the-drivers) :
1. Select the right driver given the capture card (mono, duo, quad), the driver version and the L4T version  
    sudo dpkg -i stereolabs-zedx_X.X.X-ZED-LINK-YYYY-L4TZZ.Z_arm64.deb   
    X.X.X is the driver version  
    YYYY is the ZED Link GMSL2 capture card model (i.e. MONO, DUO, QUAD)  
    L4TZZ.Z is the Jetson Linux (L4T) version  
2. sudo apt install libqt5core5a
3. sudo reboot
4. Check the installation with : sudo dmesg | grep zedx  
    -> if it outputs something, good to go  
5. (if camera has been unplugged, configuration has changed) : sudo systemctl restart zed_x_daemon  


# Setup zed drivers (https://www.stereolabs.com/en-fr/blog/getting-started-with-jetson-agx-orin-devkit) :
1. check out L4T driver : cat /etc/nv_tegra_release  
output example : # R35 (release), REVISION: 4.1, GCID: 33958178, BOARD: t186ref, EABI: aarch64, DATE: Tue Aug  1 19:57:35 UTC 2023   
-> means that the L4T version is 35.X according to the release
2. Download the driver release corresponding to your configuration (https://www.stereolabs.com/en-fr/developers/release) : In our case, it was ZED_SDK_Tegra_L4T35.4_v4.1.3.zstd.run  
3. cd ~/Downloads # replace with the correct folder if required  
   chmod +x ZED_SDK*  
   ./ZED_SDK_Tegra_L4T<l4t_version>_v<ZED_SDK_version>.run  
4. Follow the installation procedure  

# Test the camera with zed-sdk :
1. git clone https://github.com/stereolabs/zed-sdk.git
2. go to tutorials/tutorial 1 - hello ZED/python/
3. python3 python3 hello_zed.py -> should output the serial number of the camera

# Ros Wrapper for Zed camera :
1. git clone https://github.com/stereolabs/zed-ros-wrapper
2. roslaunch zed_wrapper zedx.launch  
