### Methods from Pytracking

#### PyTracking ModelZoo

The base tracker models of methods from pytracking can be downloaded [here](https://github.com/visionml/pytracking/blob/master/MODEL_ZOO.md#Models)  
We recommend download the models into `pytracking/networks`

You can run the following script for downloading the models

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


### The Methods from pysot
To introduce SiamRPN++ as a base tracker, we include [pysot](https://github.com/STVIR/pysot) into our project
```bash
echo "********* Some additional packages may be required *********"
pip install pyyaml yacs tqdm colorama matplotlib cython tensorboardX

echo "****************** Setting up environment ******************"
cd pysot
python setup.py build_ext --inplace
```


