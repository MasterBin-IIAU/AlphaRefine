## RT-MDNet: Real-Time Multi-Domain Convolutional Neural Network Tracker

Created by [Ilchae Jung](http://cvlab.postech.ac.kr/~chey0313), [Jeany Son](http://cvlab.postech.ac.kr/~jeany), [Mooyeol Baek](http://cvlab.postech.ac.kr/~mooyeol), and [Bohyung Han](http://cvlab.snu.ac.kr/~bhhan) 

### Introduction
RT-MDNet is the real-time extension of [MDNet](http://cvlab.postech.ac.kr/research/mdnet/) and is the state-of-the-art real-time tracker.
Detailed description of the system is provided by our [project page](http://cvlab.postech.ac.kr/~chey0313/real_time_mdnet/) and [paper](https://arxiv.org/pdf/1808.08834.pdf)

### Citation
If you're using this code in a publication, please cite our paper.

	@InProceedings{rtmdnet,
	author = {Jung, Ilchae and Son, Jeany and Baek, Mooyeol and Han, Bohyung},
	title = {Real-Time MDNet},
	booktitle = {European Conference on Computer Vision (ECCV)},
	month = {Sept},
	year = {2018}
	}
  
### Notice
We re-write the implementation of the roi_align to support the high version pytorch and now this code supports pytorch 1.0+.

### How to use
You need to complie the roi_align first.

**Install**
```
1. cd RT-MDNet/modules/roi_align
2. python setup.py build_ext --inplace
3. pip install sklearn
```

### System Requirements

This code is tested on 64 bit Linux (Ubuntu 16.04 LTS).

**Prerequisites** 
* Python3
* PyTorch (>= 0.4.0)
* For GPU support, a GPU (~2GB memory for test) and CUDA toolkit.
* Training Dataset (ImageNet-Vid) if needed.
  
### Online Tracking

**Pretrained Model and results**
If you only run the tracker, you can use the pretrained model: 
[RT-MDNet-ImageNet-pretrained](https://www.dropbox.com/s/lr8uft05zlo21an/rt-mdnet.pth?dl=0).
Also, results from pretrained model are provided in [here](https://www.dropbox.com/s/pefp4dqjwjows3z/RT-MDNet%20Results.zip?dl=0).

**Demo**
   0. Run 'Run.py'.

### Learning RT-MDNet
**Preparing Datasets**
1. If you download ImageNet-Vid dataset, you run 'modules/prepro_data_imagenet.py' to parse meta-data from dataset. After that, 'imagenet_refine.pkl' is generized.
2. type the path of 'imagenet_refine.pkl' in 'train_mrcnn.py'
  
**Demo**
* Run 'train_mrcnn.py' after hyper-parameter tuning suitable to the capacity of your system.
  
