# Alpha-Refine

This is the official implementation of [Alpha-Refine: Boosting Tracking Performance by Precise Bounding Box Estimation
](https://arxiv.org/abs/2012.06815).
![Architecture](doc/asset/AR-Architecture.png)

## News
- :warning: We provide a concise script [demo.py](demo.py) as an example of applying alpha refine to dimp. 
**We recommend taking this script as the starting point of exploring our project**.
- A TensorRT optimized version of AlphaRefine is available [here](https://github.com/ymzis69/AlphaRefine_TensorRT).
- The code for **CVPR2021** is updated. The old version is still available by
        
        git clone -b vot2020 https://github.com/MasterBin-IIAU/AlphaRefine.git 
        
- AlphaRefine is accepted by the **CVPR2021**
- :trophy: **Alpha-Refine wins VOT2020 Real-Time Challenge with EAOMultistart 0.499!** 
- VOT2020 winner presentation [slide](VOT20-RT-Report.pdf) has been uploaded.


## Setup Alpha-Refine

* **Install AlphaRefine**
  
```bash
git clone https://github.com/MasterBin-IIAU/AlphaRefine.git
cd AlphaRefine
```
Run the installation script to install all the dependencies. You need to provide the `${conda_install_path}`
(e.g. `~/anaconda3`) and the name `${env_name}` for the created conda environment (e.g. `alpha`).
```
# install dependencies
bash install.sh ${conda_install_path} ${env_name}
conda activate alpha
python setup.py develop
```  

* **Download AlphaRefine Models**

We provide the models of *AlphaRefine* here. The **AUC** and **Latency** are tested with SiamRPN++ as the base tracker
on *LaSOT* dataset, using a RTX 2080Ti GPU.

We recommend download the model into `ltr/checkpoints/ltr/SEx_beta`. 

| Tracker        | Backbone         | Latency     | AUC(%)   |  Model  |
|:--------------:|:----------------:|:-----------:|:-----------:|:----------------:|
| AR34<sub>c+m</sub> | ResNet34     |  5.1ms  |  55.9  |   [google](https://drive.google.com/file/d/1drLqNq4r9g4ZqGtOGuuLCmHJDh20Fu1m/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1ZCJKk1mXE_96BEpwGiEuMQ)[key:jl1m]|
| AR18<sub>c+m</sub> | ResNet18     |  4.2ms  |  55.0  |   [google](https://drive.google.com/file/d/1ANf0KCvlFBbGQPpvT-3WNiy414ANkgLZ/view?usp=sharing)/[baidu](https://pan.baidu.com/s/1IIaRNkFVPSG1s71g255CHw)[key:83ef]|

When combined with more powerful base trackers, *AlphaRefine* leads to very competitive tracking systems (e.g. *ARDiMP*). 
Following are some of the best performed trackers on LaSOT. Results are present in [Performance](#performance)

* **Demo**

We provide a concise [demo.py](demo.py) as an example for applying alpha refine to dimp.
**We recommend you should take this script as the starting point of exploring our project**.
You may need  [doc/Reproduce.md](doc/Reproduce.md) for setting up the base trackers of our experiments.

## How to apply Alpha-Refine to Your Own Tracker
We provide a concise [demo.py](demo.py) as an example for applying alpha refine to dimp.


## How to Train Alpha-Refine
Please refer to [doc/TRAIN.md](doc/TRAIN.md) for the guidance of training Alpha-Refine.

After training, you can refer to [doc/Reproduce.md](doc/Reproduce.md) for reproducing our experiment result.

## Performance

When combined with more powerful base trackers, 
*AlphaRefine* leads to very competitive tracking systems (e.g. *ARDiMP*).
For more performance reports, please refer to our [paper](https://arxiv.org/abs/2012.06815).
**You can refer to [doc/Reproduce.md](doc/Reproduce.md) for reproducing our result.**

* **LaSOT**

     | Tracker                   | Success Score    | Speed (fps) | Paper/Code |
     |:-----------               |:----------------:|:----------------:|:----------------:|
     | ARDiMP (ours)             | 0.654  |  32 (RTX 2080Ti)  |   [Paper](https://arxiv.org/abs/2012.06815)/[Result](https://drive.google.com/file/d/1UNPwz7qP8SeBTxHF_Cw0JLmrN1jTqJJE/view?usp=sharing) |
     | Siam R-CNN (CVPR20)       | 0.648  |  5 (Tesla V100)   |   [Paper](https://arxiv.org/pdf/1911.12836.pdf)/[Code](https://github.com/VisualComputingInstitute/SiamR-CNN) |
     | DimpSuper                 | 0.631  |  39 (RTX 2080Ti)  |   [Paper](https://arxiv.org/pdf/2003.12565.pdf)/[Code](https://github.com/visionml/pytracking)  |
     | ARDiMP50 (ours)           | 0.602  |  46 (RTX 2080Ti)  |   [Paper](https://arxiv.org/abs/2012.06815)/[Result](https://drive.google.com/file/d/1wJc_-1lCxeGlqEAKd1qER1x_4bWAhujv/view?usp=sharing)  |
     | PrDiMP50 (CVPR20)         | 0.598  |  30 (Unkown GPU)  |   [Paper](https://arxiv.org/pdf/2003.12565.pdf)/[Code](https://github.com/visionml/pytracking)  |
     | LTMU (CVPR20)             | 0.572  |  13 (RTX 2080Ti)  |   [Paper](https://arxiv.org/abs/2004.00305)/[Code](https://github.com/Daikenan/LTMU) |
     | DiMP50 (ICCV19)           | 0.568  |  59 (RTX 2080Ti)  |   [Paper](https://arxiv.org/pdf/1904.07220.pdf)/[Code](https://github.com/visionml/pytracking)  |
     | Ocean (ECCV20)            | 0.560  |  25 (Tesla V100)  |   [Paper](https://arxiv.org/abs/2006.10721)/[Code](https://github.com/researchmm/TracKit) |  
     | ARSiamRPN (ours)          | 0.560  |  50 (RTX 2080Ti)  |   [Paper](https://arxiv.org/abs/2012.06815)/[Result](https://drive.google.com/file/d/1u-ou43O_RU9oRFx1UKjzeYe6e-4qnMZZ/view?usp=sharing) |  
     | SiamAttn (CVPR20)         | 0.560  |  45 (RTX 2080Ti)  |   [Paper](https://arxiv.org/pdf/2004.06711.pdf)/[Code]() |
     | SiamFC++GoogLeNet (AAAI20)| 0.544  |  90 (RTX 2080Ti)  |   [Paper](https://arxiv.org/pdf/1911.06188.pdf)/[Code](https://github.com/MegviiDetection/video_analyst) |
     | MAML-FCOS (CVPR20)        | 0.523  |  42 (NVIDIA P100) |   [Paper](https://arxiv.org/pdf/2004.00830.pdf)/[Code]() |
     | GlobalTrack (AAAI20)      | 0.521  |  6 (GTX TitanX)   |   [Paper](https://arxiv.org/abs/1912.08531)/[Code](https://github.com/huanglianghua/GlobalTrack) |
     | ATOM (CVPR19)             | 0.515  |  30 (GTX 1080)    |   [Paper](https://arxiv.org/pdf/1811.07628.pdf)/[Code](https://github.com/visionml/pytracking)  |


## Acknowledgments
* This repo is based on [Pytracking](https://github.com/visionml/pytracking.git) which is an exellent work.
* Thanks for [pysot](https://github.com/STVIR/pysot) and [RTMDNet](https://github.com/IlchaeJung/RT-MDNet) from which
 we borrow the code as base trackers.

