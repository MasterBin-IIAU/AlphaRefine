import os
######################################################################
dataset_name_ = 'LaSOT'
video_name_ = ''  # 'airplane-9'

######################################################################
dataset_root_ = '/media/zxy/Samsung_T5/Data/DataSets/LaSOT/LaSOT_Test'
save_dir = '/home/zxy/Desktop/AlphaRefine/analysis'

######################### Refine Module ################################
model_dir = '/home/zxy/Desktop/AlphaRefine/pytracking/ltr/checkpoints/ltr/'
model_code = 'a'
refine_path = os.path.join(model_dir, "SEx_beta/SEcm_r34/SEcmnet_ep0040-{}.pth.tar".format(model_code))
RF_type = 'AR_CrsM_R34SR20_pixCorr_woPr_woNL_corner_{}'.format(model_code)
selector_path = 0
sr = 2.0; input_sz = int(128 * sr)  # 2.0 by default
