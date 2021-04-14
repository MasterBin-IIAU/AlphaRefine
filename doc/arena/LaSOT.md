#### Dowload Models

#### Arrange LaSOT Dataset
In order to evaluate with the scripts, `arena/LaSOT/tools/pick_LaSOT_test.py`

##### Download AlphaRefine
We recommend download the model into `ltr/checkpoints/ltr/SEx_beta`


#### Run Experiment Scripts
We take [arena/LaSOT/run_dimp_RF.py](arena/LaSOT/run_dimp_RF.py) as an example:

##### path setting up
1. Edit [arena/LaSOT/common_path.py](arena/LaSOT/common_path.py), specify 'refine_path' as 
`ltr/checkpoints/ltr/SEx_beta/SEcm_r34/SEcmnet_ep0040-c.pth.tar`

2. specify the 'dataset_root_' as the `path/to/the/LaSOT_test_set`.

##### run the following command
```
cd arena/LaSOT
python run_dimp_RF.py
python eval.py
```