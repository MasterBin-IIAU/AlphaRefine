#!/bin/bash

if [ "$#" -ne 2 ]; then
    echo "ERROR! Illegal number of parameters. Usage: bash install.sh conda_install_path environment_name"
    exit 0
fi

conda_install_path=$1
conda_env_name=$2

source $conda_install_path/etc/profile.d/conda.sh
echo "****************** Creating conda environment ${conda_env_name} python=3.7 ******************"
conda create -y --name $conda_env_name

echo ""
echo ""
echo "****************** Activating conda environment ${conda_env_name} ******************"
conda activate $conda_env_name

echo ""
echo ""
echo "****************** Installing pytorch with cuda10 ******************"
# conda install -y pytorch torchvision cudatoolkit=10.0 -c pytorch  # the pytracking setup script install the newest version of pytroch by default, which may lead to some problems
conda install -y pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch

echo ""
echo ""
echo "****************** Installing matplotlib ******************"
conda install -y matplotlib

echo ""
echo ""
echo "****************** Installing pandas ******************"
conda install -y pandas

echo ""
echo ""
echo "****************** Installing tqdm ******************"
conda install -y tqdm

echo ""
echo ""
echo "****************** Installing opencv ******************"
pip install opencv-python

echo ""
echo ""
echo "****************** Installing tensorboard ******************"
pip install tb-nightly
pip install tensorboardX

echo ""
echo ""
echo "****************** Installing visdom ******************"
pip install visdom

echo ""
echo ""
echo "****************** Installing scikit-image ******************"
pip install scikit-image

echo ""
echo ""
echo "****************** Installing tikzplotlib ******************"
pip install tikzplotlib

echo ""
echo ""
echo "****************** Installing gdown ******************"
pip install gdown

echo ""
echo ""
echo "****************** Installing cython ******************"
conda install -y cython

echo ""
echo ""
echo "****************** Installing coco toolkit ******************"
pip install pycocotools

echo ""
echo ""
echo "****************** Installing LVIS toolkit ******************"
pip install lvis

echo ""
echo ""
echo "**************** Installing pytracking as a pkg **************"
python setup.py develop


echo ""
echo ""
echo "****************** Installing jpeg4py python wrapper ******************"
pip install jpeg4py 

echo ""
echo ""
echo "****************** Installing ninja-build to compile PreROIPooling ******************"

echo "************************* Need sudo privilege ******************"
sudo apt-get install ninja-build

echo ""
echo ""
echo "****************** Setting up environment ******************"
python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"
python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"


echo ""
echo ""
echo "****************** Installing jpeg4py ******************"
while true; do
    read -p "Install jpeg4py for reading images? This step required sudo privilege. Installing jpeg4py is optional, however recommended. [y,n]  " install_flag
    case $install_flag in
        [Yy]* ) sudo apt-get install libturbojpeg; break;;
        [Nn]* ) echo "Skipping jpeg4py installation!"; break;;
        * ) echo "Please answer y or n  ";;
    esac
done

echo ""
echo ""
echo "****************** Installation complete! ******************"

echo ""
echo ""
echo "****************** More pytracking networks can be downloaded from the google drive folder https://drive.google.com/drive/folders/1WVhJqvdu-_JG1U-V0IqfxTUa1SBPnL0O ******************"
echo "****************** Or, visit the model zoo at https://github.com/visionml/pytracking/blob/master/MODEL_ZOO.md ******************"
