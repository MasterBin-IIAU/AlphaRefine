from .common_path import *
'''Things that we can change'''
###################################################
siam_model_ = 'siamrpn_r50_l234_dwxcorr'
# siam_model_ = 'siamrpn_r50_l234_dwxcorr_otb'
# siam_model_ = 'siammask_r50_l3'
###################################################
# dataset_name_ = 'GOT10K'
# dataset_name_ = 'VOT2018'
# dataset_name_ = 'VOT2018-LT'
dataset_name_ = 'LaSOT'
###################################################
video_name_ = ''
# video_name_ = 'airplane-9'
#########################################################################################
import os
project_path_ = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../external/pysot'))
