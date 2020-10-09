# Requirements
## Hardware Requirements
* GPU: NVIDIA RTX TITAN (24GB Memory)
* CPU: Intel® Core™ i9-9900K CPU @ 3.60GHz 
* SSD: SanDisk 1TB
## Software Dependency
* OS: Ubuntu 16.04 LTS
* Anaconda 3
* CUDA 10.0, CUDNN 7.6
* Python libraries: Python3.7, Pytorch 1.2.0
# Installation

This document contains detailed instructions for installing the necessary dependencies for **AlphaRefine**. 
The instrustions have been tested on an Ubuntu 16.04 system.  
 
### AlphaRefine-Related Part
* Create and activate a conda environment
```bash
conda create -n alpharefine python=3.7
conda activate alpharefine
```

* Install PyTorch  
```bash
conda install pytorch==1.2.0 torchvision==0.4.0 cudatoolkit=10.0 -c pytorch
```

* Install matplotlib, pandas, tqdm, opencv, scikit-image, visdom, easydict and tikzplotlib 
```bash
conda install matplotlib pandas tqdm
pip install opencv-python visdom scikit-image easydict tikzplotlib
```


* Install ninja-build for Precise ROI pooling  
```bash
sudo apt-get install ninja-build
```
In case of issues, we refer to https://github.com/vacancy/PreciseRoIPooling.  


* Install jpeg4py  
```bash
pip install jpeg4py 
```


* Setup the environment  
Create the default environment setting files. 
```bash
# Change directory to <PATH_of_AlphaRefine_submit>
cd AlphaRefine_submit

# Environment settings for pytracking. Saved at pytracking/evaluation/local.py
python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"

# Environment settings for ltr. Saved at ltr/admin/local.py
python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"
```

You can modify these files to set the paths to datasets, results paths etc.  
* Add the project path to environment variables  
Open ~/.bashrc, and add the following line to the end. Note to change <path_of_AlphaRefine_submit> to your real path.
```
export PYTHONPATH=<path_of_AlphaRefine_submit>:$PYTHONPATH
```

* Download the pre-trained networks  
Step1: Download the network for [super_dimp](https://drive.google.com/file/d/1qDptswis2FxihLRYLVRGDvx6aUoAVVLv/view), 
and put it in the directory set by "network_path" in "pytracking/evaluation/local.py". By default, it is set to 
pytracking/networks.  
Step2: Download the network for [Alpha-Refine](https://drive.google.com/open?id=1qOQRfaRMbQ2nmgX1NFjoQHfXOAn609QM) 
and put it under the ltr/checkpoints/ltr/ARcm_seg/ARcm_coco_seg_only_mask_384 dir.
### VOT-Toolkit-Related Part
* Install vot-toolkit in the current conda environment
```bash
pip install git+https://github.com/votchallenge/vot-toolkit-python
```
* Reinstall pillow to avoid "ImportError: cannot import name 'PILLOW_VERSION' from 'PIL'"
```
conda install pillow=6.1
```
* Modify [paths](https://github.com/MasterBin-IIAU/AlphaRefine/blob/46bec7318702090199a49306ff3d597c2fc140f9/AlphaRefine_submit/AlphaRefine/trackers.ini#L6) to the real path on your machine. 
# Evaluation
* Go to AlphaRefine dir and modify some paths  
Modify paths in 'trackers.ini' and 'exp.sh' to your real paths.

* Run AlphaRefine and analyze results  
**Note**: After download VOT2020-ST dataset, please move it to SSD for imrpoving I/O speed :)  
You can create a soft link called 'sequences' and put in under 'AlphaRefine' dir.
```bash
sh exp.sh
```
