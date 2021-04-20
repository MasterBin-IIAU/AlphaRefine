#### Download AlphaRefine Models
We provide the models of *AlphaRefine* here. The **AUC** and **Latency** are tested with SiamRPN++ as the base tracker
on *LaSOT* dataset, using a RTX 2080Ti GPU.

We recommend download the model into `ltr/checkpoints/ltr/SEx_beta`. 

| Tracker        | Backbone         | Latency     | AUC(%)   |  Model  |
|:--------------:|:----------------:|:-----------:|:-----------:|:----------------:|
| AR34<sub>c+m</sub> | ResNet34     |  5.1ms  |  55.9  |   [model](https://drive.google.com/file/d/1drLqNq4r9g4ZqGtOGuuLCmHJDh20Fu1m/view?usp=sharing)|
| AR18<sub>c+m</sub> | ResNet18     |  4.2ms  |  55.0  |   [model](https://drive.google.com/file/d/1ANf0KCvlFBbGQPpvT-3WNiy414ANkgLZ/view?usp=sharing)|

When combined with more powerful base trackers, 
*AlphaRefine* leads to very competitive tracking systems (e.g. *ARDiMP*). 
Following are some of the best performed trackers on LaSOT. 
More results are present in [Performance](#performance)

| Tracker                   | AUC(%)    | Speed (fps) | Paper/Code |
|:-----------               |:----------------:|:----------------:|:----------------:|
| ARDiMP (ours)             | 65.4  |  32 (RTX 2080Ti)  |   [Paper](https://arxiv.org/abs/2012.06815)/[Result](https://drive.google.com/file/d/1UNPwz7qP8SeBTxHF_Cw0JLmrN1jTqJJE/view?usp=sharing) |
| Siam R-CNN (CVPR20)       | 64.8  |  5 (Tesla V100)   |   [Paper](https://arxiv.org/pdf/1911.12836.pdf)/[Code](https://github.com/VisualComputingInstitute/SiamR-CNN) |
| DimpSuper                 | 63.1  |  39 (RTX 2080Ti)  |   [Paper](https://arxiv.org/pdf/2003.12565.pdf)/[Code](https://github.com/visionml/pytracking)  |
| ARDiMP50 (ours)           | 60.2  |  46 (RTX 2080Ti)  |   [Paper](https://arxiv.org/abs/2012.06815)/[Result](https://drive.google.com/file/d/1wJc_-1lCxeGlqEAKd1qER1x_4bWAhujv/view?usp=sharing)  |
| PrDiMP50 (CVPR20)         | 59.8  |  30 (Unkown GPU)  |   [Paper](https://arxiv.org/pdf/2003.12565.pdf)/[Code](https://github.com/visionml/pytracking)  |
| LTMU (CVPR20)             | 57.2  |  13 (RTX 2080Ti)  |   [Paper](https://arxiv.org/abs/2004.00305)/[Code](https://github.com/Daikenan/LTMU) |



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

##### run the inference script
```
cd arena/LaSOT
python run_dimp_RF.py
```
The result files will be saved in `arena/LaSOT/analysis`
##### evaluate the result files
```
# evaluate all the results under 'analysis/'
python eval.py

# using --tracker_prefix to specify the result to evaluate
python eval.py --tracker_prefix ${exp_name}
```