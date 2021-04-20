#### Download AlphaRefine Models
We provide the models of *AlphaRefine* here. The **AUC** and **Latency** are tested with SiamRPN++ as the base tracker
on *LaSOT* dataset, using a RTX 2080Ti GPU.

We recommend download the model into `ltr/checkpoints/ltr/SEx_beta`. 

| Tracker        | Backbone         | Latency     | AUC(%)   |  Model  |
|:--------------:|:----------------:|:-----------:|:-----------:|:----------------:|
| AR34<sub>c+m</sub> | ResNet34     |  5.1ms  |  55.9  |   [model](https://drive.google.com/file/d/1drLqNq4r9g4ZqGtOGuuLCmHJDh20Fu1m/view?usp=sharing)|
| AR18<sub>c+m</sub> | ResNet18     |  4.2ms  |  55.0  |   [model](https://drive.google.com/file/d/1ANf0KCvlFBbGQPpvT-3WNiy414ANkgLZ/view?usp=sharing)|


##### Download AlphaRefine
We recommend download the model into `ltr/checkpoints/ltr/SEx_beta`

#### Run Experiment Scripts
We take [arena/TrackingNet/run_trackingnet_pytracking_RF.py](arena/TrackingNet/run_trackingnet_pytracking_RF.py) as an example:

##### path setting up
1. Edit [arena/TrackingNet/common_path.py](arena/TrackingNet/common_path.py), specify 'refine_path' as 
`ltr/checkpoints/ltr/SEx_beta/SEcm_r34/SEcmnet_ep0040-c.pth.tar`

2. Check if you have specified the 'dataset_root_' as the `path/to/the/TrackingNet/TEST`.

##### run the inference script
```
cd arena/TrackingNet
python run_trackingnet_pytracking_RF.py
```
The result files will be saved in `arena/trackingnet/analysis`

##### evaluate the result files
upload the zipped result to TrackingNet submitting website 
[EvalAI](http://eval.tracking-net.org/web/challenges/challenge-page/39/submission)
