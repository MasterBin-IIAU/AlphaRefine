## Reproduce Our Experiment Results

### Establish Base Trackers
In this project, we introduce DiMP50, DiMPsuper, ATOM, ECO, RTMDNet, SiamRPN++ as our base trackers.

#### PyTracking Methods
DiMP50, DiMPsuper, ATOM, ECO are trackers from [PyTracking](pytracking).

The base tracker models trained using PyTracking can be download from [model zoo](https://github.com/visionml/pytracking/blob/master/MODEL_ZOO.md), download them into `pytracking/networks` 

Or you can run the following script to download the models.

```
echo "****************** Downloading networks ******************"
mkdir pytracking/networks

echo "****************** DiMP Network ******************"
gdown https://drive.google.com/uc\?id\=1qgachgqks2UGjKx-GdO1qylBDdB1f9KN -O pytracking/networks/dimp50.pth
gdown https://drive.google.com/uc\?id\=1MAjrRJDCbL0DSjUKFyDkUuYS1-cYBNjk -O pytracking/networks/dimp18.pth
gdown https://drive.google.com/open?id=1qDptswis2FxihLRYLVRGDvx6aUoAVVLv -O pytracking/networks/super_dimp.pth

echo "****************** ATOM Network ******************"
gdown https://drive.google.com/uc\?id\=1VNyr-Ds0khjM0zaq6lU-xfY74-iWxBvU -O pytracking/networks/atom_default.pth

echo "****************** ECO Network ******************"
gdown https://drive.google.com/uc\?id\=1aWC4waLv_te-BULoy0k-n_zS-ONms21S -O pytracking/networks/resnet18_vggmconv1.pth
```

#### [Optional] Other Base Trackers
Please refer to [external/pysot/README.md](external/pysot/README.md) for establishing SiamRPN++ and
[external/RT_MDNet/README.md](external/RT_MDNet/README.md) for establishing RTMDNet.


### Run Evaluation Scripts

* We provide the evaluation recipes of [LaSOT](doc/arena/LaSOT.md) | [GOT-10K](doc/arena/GOT-10K.md) | 
[TrackingNet](doc/arena/TrackingNet.md) | [VOT2020](doc/arena/VOT2020.md).
    You can follow these recipes to run the evaluation scripts.

* For some of the testing scripts, the path to the testing sets should be specified in `pytracking/evaluation/local.py`
    
    If `pytracking/evaluation/local.py` is not exist, please run
    ```
    python -c "from pytracking.evaluation.environment import create_default_local_file; create_default_local_file()"
    ```
    An example of `pytracking/evaluation/local.py.example` is provided.
