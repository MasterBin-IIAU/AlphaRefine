#### Dowload Models

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
