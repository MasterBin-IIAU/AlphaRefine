import os
######################################################################
_dataset_name = 'GOT-10k'
_video_name = ''  # 'airplane-9'

######################################################################
_dataset_root = '/home/zxy/Downloads/Datasets'
save_dir = '/home/zxy/Desktop/AlphaRefine/analysis'

######################### Refine Module ################################
model_dir = '/home/zxy/Desktop/AlphaRefine/pytracking/ltr/checkpoints/ltr/'
refine_path = os.path.join(model_dir, "SEx_beta/SEcm_r34/SEcmnet_ep0040-{}.pth.tar")
RF_type = 'AR_CrsM_R34SR20_pixCorr_woPr_woNL_corner_{}'
selector_path = 0
sr = 2.0; input_sz = int(128 * sr)  # 2.0 by default
