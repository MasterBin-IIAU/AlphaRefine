#### Dowload Models

#### Arrange LaSOT Dataset
We borrow the LaSOT evaluation code from [pysot](https://github.com/StrangerZhang/pysot-toolkit). 
The following procedure arrange the LaSOT dataset into the required structure.

1. Specify the path to the LaSOT data `LaSOT_root_dir` and the dumping path `LaSOT_test_dir`
in `arena/LaSOT/tools/pick_LaSOT_test.py`.

2. run `arena/LaSOT/tools/pick_LaSOT_test.py` to extract the testing sequence from LaSOTBenchmark.
 Download the file index [here](https://drive.google.com/drive/folders/10cfXjwQQBQeu48XMf2xc_W1LucpistPI)
 and put it into the `LaSOT_test_dir`

3. Specify `dataset_root_` in `AlphaRefine/arena/LaSOT/common_path.py` as the `LaSOT_test_dir`

##### Download AlphaRefine
We recommend download the model into `ltr/checkpoints/ltr/SEx_beta`


#### Run Experiment Scripts
We take [arena/LaSOT/run_dimp_RF.py](arena/LaSOT/run_dimp_RF.py) as an example:

##### path setting up
1. Edit [arena/LaSOT/common_path.py](arena/LaSOT/common_path.py), specify 'refine_path' as 
`ltr/checkpoints/ltr/SEx_beta/SEcm_r34/SEcmnet_ep0040-c.pth.tar`

2. Check if you have specified the 'dataset_root_' as the `path/to/the/LaSOT_test`.

##### run the following command
```
cd arena/LaSOT
python run_dimp_RF.py
python eval.py
```