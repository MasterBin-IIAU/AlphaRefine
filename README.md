# Alpha-Refine
This is the official implementation of [Alpha-Refine: Boosting Tracking Performance by Precise Bounding Box Estimation
](https://arxiv.org/abs/2012.06815)

A more detailed document is on its way.
![Architecture](doc/asset/AR-Architecture.png)
## Getting Start

#### Install dependencies
Run the installation script to install all the dependencies. You need to provide the conda install path (e.g. ~/anaconda3) and the name for the created conda environment (here ```pytracking```).  
```bash
bash install.sh conda_install_path alpha
conda activate alpha
```  

#### Download Models
The final version of our model is available [here](https://drive.google.com/file/d/1drLqNq4r9g4ZqGtOGuuLCmHJDh20Fu1m/view?usp=sharing),
download it into `ltr/checkpoints/ltr/SEx_beta/SEcm_r34`

The base tracker models trained using PyTracking in the [model zoo](MODEL_ZOO.md), download them into `pytracking/networks` 

#### Run Experiment Scripts
We take [arena/LaSOT/run_dimp_RF.py](arena/LaSOT/run_dimp_RF.py) as an example:

##### path setting up
1. Edit [arena/LaSOT/common_path.py](arena/LaSOT/common_path.py), specify 'refine_path' as 
`ltr/checkpoints/ltr/SEx_beta/SEcm_r34/SEcmnet_ep0040-c.pth.tar `

2. specify the 'dataset_root_' as the `path/to/the/LaSOT_test_set`.

##### run the following command
```
cd arena/LaSOT
python run_dimp_RF.py
python eval.py
```

#### Other Base Trackers
Please refer to [pysot/README.md](pysot/README.md) and [RT_MDNet/README.md](RT_MDNet/README.md)


#### Install dependencies
Run the installation script to install all the dependencies. You need to provide the conda install path (e.g. ~/anaconda3) and the name for the created conda environment (here ```pytracking```).  
```bash
bash install.sh conda_install_path alpha
conda activate alpha
```  

## Train 

The training code is based on [Pytracking](https://github.com/visionml/pytracking.git), thus the training operation is similar.

#### Setting Up the Datasets
The training recipe are placed in `ltr/train_settings` (e.g. `ltr/train_settings/SEx_beta/SEcm_r34.py`), you can
configure the *training parameters* and *Dataloaders*. The `path/to/data` should be specified in `ltr/admin/local.py` to make
the *Dataloaders* find the data.

#### Run Training Scripts
For the recipe `ltr/train_settings/$sub1/$sub2.py` run the following command to launch the training procedure.
```
python -m torch.distributed.launch --nproc_per_node=8 \
        run_training_multigpu.py $sub1 $sub2 
```
The checkpoints will be saved in `ltr/checkpoints/ltr/$sub1/$sub2/SEcmnet_ep00*.pth.tar`.


## Performance

* **LaSOT**

     | Tracker                   | Success Score    | Speed (fps) | Paper/Code |
     |:-----------               |:----------------:|:----------------:|:----------------:|
     | ARDiMP (ours)             | 0.654  |  32 (RTX 2080Ti)  |   [Paper](https://arxiv.org/abs/2012.06815)/This Repo/[Result](https://drive.google.com/file/d/1UNPwz7qP8SeBTxHF_Cw0JLmrN1jTqJJE/view?usp=sharing) |
     | Siam R-CNN (CVPR20)       | 0.648  |  5 (Tesla V100)   |   [Paper](https://arxiv.org/pdf/1911.12836.pdf)/[Code](https://github.com/VisualComputingInstitute/SiamR-CNN) |
     | DimpSuper                 | 0.631  |  39 (RTX 2080Ti)  |   [Paper](https://arxiv.org/pdf/2003.12565.pdf)/[Code](https://github.com/visionml/pytracking)  |
     | ARDiMP50 (ours)           | 0.602  |  46 (RTX 2080Ti)  |   [Paper](https://arxiv.org/abs/2012.06815)/This Repo/[Result](https://drive.google.com/file/d/1wJc_-1lCxeGlqEAKd1qER1x_4bWAhujv/view?usp=sharing)  |
     | PrDiMP50 (CVPR20)         | 0.598  |  30 (Unkown GPU)  |   [Paper](https://arxiv.org/pdf/2003.12565.pdf)/[Code](https://github.com/visionml/pytracking)  |
     | LTMU (CVPR20)             | 0.572  |  13 (RTX 2080Ti)  |   [Paper](https://arxiv.org/abs/2004.00305)/[Code](https://github.com/Daikenan/LTMU) |
     | DiMP50 (ICCV19)           | 0.568  |  59 (RTX 2080Ti)  |   [Paper](https://arxiv.org/pdf/1904.07220.pdf)/[Code](https://github.com/visionml/pytracking)  |
     | Ocean (ECCV20)            | 0.560  |  25 (Tesla V100)  |   [Paper](https://arxiv.org/abs/2006.10721)/[Code](https://github.com/researchmm/TracKit) |  
     | ARSiamRPN (ours)          | 0.560  |  50 (RTX 2080Ti)  |   [Paper](https://arxiv.org/abs/2012.06815)/This Repo/[Result](https://drive.google.com/file/d/1u-ou43O_RU9oRFx1UKjzeYe6e-4qnMZZ/view?usp=sharing) |  
     | SiamAttn (CVPR20)         | 0.560  |  45 (RTX 2080Ti)  |   [Paper](https://arxiv.org/pdf/2004.06711.pdf)/[Code]() |
     | SiamFC++GoogLeNet (AAAI20)| 0.544  |  90 (RTX 2080Ti)  |   [Paper](https://arxiv.org/pdf/1911.06188.pdf)/[Code](https://github.com/MegviiDetection/video_analyst) |
     | MAML-FCOS (CVPR20)        | 0.523  |  42 (NVIDIA P100) |   [Paper](https://arxiv.org/pdf/2004.00830.pdf)/[Code]() |
     | GlobalTrack (AAAI20)      | 0.521  |  6 (GTX TitanX)   |   [Paper](https://arxiv.org/abs/1912.08531)/[Code](https://github.com/huanglianghua/GlobalTrack) |
     | ATOM (CVPR19)             | 0.515  |  30 (GTX 1080)    |   [Paper](https://arxiv.org/pdf/1811.07628.pdf)/[Code](https://github.com/visionml/pytracking)  |




## Alpha-Refine is Based on PyTracking Code Base
PyTracking is a general python framework for visual object tracking and video object segmentation,
based on **PyTorch**.


### Base Trackers
The toolkit contains the implementation of the following trackers.  

##### PrDiMP
**[[Paper]](https://arxiv.org/pdf/2003.12565)  [[Raw results]](MODEL_ZOO.md#Raw-Results)
  [[Models]](MODEL_ZOO.md#Models)  [[Training Code]](./ltr/README.md#PrDiMP)  [[Tracker Code]](./pytracking/README.md#DiMP)**
    

##### DiMP
**[[Paper]](https://arxiv.org/pdf/1904.07220)  [[Raw results]](MODEL_ZOO.md#Raw-Results)
  [[Models]](MODEL_ZOO.md#Models)  [[Training Code]](./ltr/README.md#DiMP)  [[Tracker Code]](./pytracking/README.md#DiMP)**
    
 
##### ATOM
**[[Paper]](https://arxiv.org/pdf/1811.07628)  [[Raw results]](MODEL_ZOO.md#Raw-Results)
  [[Models]](MODEL_ZOO.md#Models)  [[Training Code]](./ltr/README.md#ATOM)  [[Tracker Code]](./pytracking/README.md#ATOM)**  

 
##### ECO
**[[Paper]](https://arxiv.org/pdf/1611.09224.pdf)  [[Models]](https://drive.google.com/open?id=1aWC4waLv_te-BULoy0k-n_zS-ONms21S)  [[Tracker Code]](./pytracking/README.md#ECO)**  


## Acknowledgments
* This repo is based on [Pytracking](https://github.com/visionml/pytracking.git) which is an exellent work.
* Thansk for [pysot](https://github.com/STVIR/pysot) and [RTMDNet](https://github.com/IlchaeJung/RT-MDNet) from which we
 we borrow the code as base trackers.

