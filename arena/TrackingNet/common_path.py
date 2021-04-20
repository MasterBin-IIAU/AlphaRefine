import os

######################################################################
dataset_name_ = 'LaSOT'
video_name_ = ''  # 'airplane-9'

######################################################################
dataset_root_ = '/home/zxy/Downloads/AR_Data/TrackingNet/TEST'
save_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../analysis'))

######################### Refine Module ################################
# model_code = 'd'
# refine_path = "/home/zxy/Desktop/AlphaRefine/pytracking/ltr/checkpoints/ltr/SEx_beta/SEcm_r34/SEcmnet_ep0040-{}.pth.tar".format(model_code)
# RF_type = 'RF_CrsM_woPr_R34SR20_{}'.format(model_code)

refine_path = '/home/zxy/Desktop/AR_Maintaince/AlphaRefine/ltr/checkpoints/ltr/SEx_beta/SEcmnet_ep0040-c.pth.tar'
RF_type = 'AR_CrsM_R34SR20_pixCorr_woPr_woNL_corner'
selector_path = 0
sr = 2.0; input_sz = int(128 * sr)  # 2.0 by default
