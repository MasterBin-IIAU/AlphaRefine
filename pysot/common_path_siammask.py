import os
# from pytracking.Refine_module_bcm import Refine_module_bcm
'''Things that we can change'''
###################################################
siam_model_ = 'siammask_r50_l3'
###################################################
# dataset_name_ = 'OTB100'
# dataset_name_ = 'GOT10K'
dataset_name_ = 'VOT2018'
# dataset_name_ = 'VOT2018-LT'
# dataset_name_ = 'LaSOT'
###################################################
video_name_ = ''
# video_name_ = 'airplane-13'
# video_name_ = 'rabbit-19'
# video_name_ = 'cattle-12'
# video_name_ = 'guitar-10'
# video_name_ = 'airplane-1'
# video_name_ = 'drone-2'
# video_name_ = 'monkey-3'
#########################################################################################
project_path_ = '/home/masterbin-iiau/Object_tracking/pysot'
dataset_root_ = '/home/masterbin-iiau/Desktop/tracking_datasets'
'''Pysot'''
snapshot_path_ = os.path.join(project_path_,'experiments/%s/model.pth'%siam_model_)
config_path_ = os.path.join(project_path_,'experiments/%s/config.yaml'%siam_model_)
'''Refinement Module'''
project_path = '/home/masterbin-iiau/Desktop/scale-estimator'
refine_root = os.path.join(project_path, 'ltr/checkpoints/ltr/SEbcm/')
refine_model_name = 'SEbcm'
refine_path = os.path.join(refine_root, refine_model_name)
selector_root = os.path.join(project_path, 'ltr/checkpoints/ltr/selector')
selector_model_name = 'selector_bcm'
selector_path = os.path.join(selector_root, selector_model_name)

'''Dataset'''
if dataset_name_ == 'GOT10K':
    dataset_dir_ = os.path.join(dataset_root_, dataset_name_,'train')
else:
    dataset_dir_ = os.path.join(dataset_root_,dataset_name_)