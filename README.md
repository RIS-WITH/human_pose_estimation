# human_pose_estimation

## Install on ROS-Noetic

1. git clone branch dev  
2. pip install requirements.txt
3. (optional) if you have the models on your machine, copy them into the models folder, else they will be downloaded through ultralytics upon launch

## Install on Jetson ORIN with Python 3.8 (JetPack 5.x)

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

## Launch

1. roslaunch human_pose_estimation rs_d435i.launch
2. roslaunch human_pose_estimation human_pose_estimation.launch