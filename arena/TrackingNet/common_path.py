######################################################################
dataset_name_ = 'LaSOT'
video_name_ = ''  # 'airplane-9'

######################################################################
dataset_root_ = '/home/zxy/SSD/zxy/TrackingNet/TEST'
save_dir = '/home/zxy/Desktop/AlphaRefine/analysis'

######################### Refine Module ################################
model_code = 'd'
refine_path = "/home/zxy/Desktop/AlphaRefine/pytracking/ltr/checkpoints/ltr/SEx_beta/SEcm_r34/SEcmnet_ep0040-{}.pth.tar".format(model_code)
RF_type = 'RF_CrsM_woPr_R34SR20_{}'.format(model_code)
selector_path = 0
sr = 2.0; input_sz = int(128 * sr)  # 2.0 by default
