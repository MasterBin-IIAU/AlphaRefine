def main():
    ''' refinement module testing code '''
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    save_root = '/home/masterbin-iiau/Desktop/scale_estimator_debug'
    project_path = '/home/masterbin-iiau/Desktop/scale-estimator'
    refine_root = os.path.join(project_path, 'ltr/checkpoints/ltr/SEbcm/')
    refine_model_name = 'SEbcm'
    refine_path = os.path.join(refine_root, refine_model_name)
    selector_root = os.path.join(project_path, 'ltr/checkpoints/ltr/selector')
    selector_model_name = 'selector_bcm'
    selector_path = os.path.join(selector_root, selector_model_name)
    SE_module = RefineModule(refine_path, selector_path)
    video_dir = '/media/masterbin-iiau/WIN_SSD/GOT10K/train/GOT-10k_Train_008936'
    # video_dir = '/media/masterbin-iiau/WIN_SSD/GOT10K/test/GOT-10k_Test_000001'
    # video_dir = '/media/masterbin-iiau/EAGET-USB/LaSOT_Test/airplane-1'
    # video_dir = '/media/masterbin-iiau/EAGET-USB/LaSOTBenchmark/airplane/airplane-9'
    video_name = video_dir.split('/')[-1]
    save_dir = os.path.join(save_root, video_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    gt_file = os.path.join(video_dir, 'groundtruth.txt')
    gt = np.loadtxt(gt_file, dtype=np.float32, delimiter=',')
    frame1_path = os.path.join(video_dir, '00000001.jpg')
    # frame1_path = os.path.join(video_dir, 'img','00000001.jpg')
    frame1 = cv2.cvtColor(cv2.imread(frame1_path), cv2.COLOR_BGR2RGB)
    SE_module.initialize(frame1, gt[0])
    for i in range(1, gt.shape[0]):
        # print(i)
        # frame_path = os.path.join(video_dir,'img', '%08d.jpg'%(i+1))
        frame_path = os.path.join(video_dir, '%08d.jpg' % (i + 1))
        frame = cv2.imread(frame_path)
        frame_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        output_dict = SE_module.refine(frame_RGB, gt[i], mode='all', test=True)
        '''add bbox'''
        # frame = add_frame_bbox(frame,output_dict['bbox'],(255,0,0))
        '''add mask'''
        frame = add_frame_mask(frame, output_dict['mask'], 0.5)
        '''add mask bbox'''
        # frame = add_frame_bbox(frame,output_dict['mask_bbox'],(0,0,255))
        '''add corner'''
        # frame = add_frame_bbox(frame,output_dict['corner'],(0,255,0))
        '''add fuse bbox'''
        frame = add_frame_bbox(frame, output_dict['all'], (0, 0, 255))
        '''show'''
        save_path = os.path.join(save_dir, '%08d.jpg' % (i + 1))
        cv2.imwrite(save_path, frame)


if __name__ == '__main__':
    main()
