## Train
The training code is based on [Pytracking](https://github.com/visionml/pytracking.git), thus the training operation is similar.

### Dataset

* Download the Dataset
    [GOT-10K](http://got-10k.aitestunion.com/downloads) |
    [LaSOT](http://vision.cs.stonybrook.edu/~lasot/download.html) |
    [MS-COCO](http://cocodataset.org/#home) |
    [ILSVRC-VID](http://image-net.org/challenges/LSVRC/2017/) |
    [ImageNet-DET](http://image-net.org/challenges/LSVRC/2017/) |
    [YouTube-VOS](https://youtube-vos.org) |
    [Saliency](https://drive.google.com/file/d/1XykS4zJl249PuyKu7aHshB5d6Y-F2k0p/view)
    
    For more details, you can refer to [ltr/README.md](https://github.com/visionml/pytracking/tree/master/ltr#overview)
    

* The path to the training sets should be specified in `ltr/admin/local.py`
    
    If the `ltr/admin/local.py` is not exist, please run 
    ``` bash
    python -c "from ltr.admin.environment import create_default_local_file; create_default_local_file()"
    ```
    An example `ltr/admin/local.py.example` is also provided.
    

### Run Training Scripts

The training recipes are placed in `ltr/train_settings` (e.g. `ltr/train_settings/SEx_beta/SEcm_r34.py`), you can
configure the *training parameters* and *Dataloaders*. 

For a recipe named `ltr/train_settings/$sub1/$sub2.py`, run the following command to launch the training procedure.
```
python -m torch.distributed.launch --nproc_per_node=8 \
        run_training_multigpu.py $sub1 $sub2 
```
The checkpoints will be saved in `AlphaRefine/checkpoints/ltr/$sub1/$sub2/SEcmnet_ep00*.pth.tar`.
