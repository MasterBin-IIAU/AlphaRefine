import os

save_root = os.path.join('/home/alphabin/Desktop/AlphaRefine_submit/vot20_debug')

# ========================= Refine Module =========================
model_dir = '/home/zxy/Desktop/AlphaRefine/pytracking/ltr/checkpoints/ltr/'
model_code = 'a'
refine_path = os.path.join(model_dir, "SEx_beta/SEcm_r34/SEcmnet_ep0040-{}.pth.tar".format(model_code))
RF_type = 'AR_CrsM_R34SR20_pixCorr_woPr_woNL_corner_{}'.format(model_code)
sr = 2.0; input_sz = int(128 * sr)  # 2.0 by default

# ========================= Base Tracker=========================
_tracker_name = 'atom'
_tracker_param = 'default'
_mask_thres = 0.65