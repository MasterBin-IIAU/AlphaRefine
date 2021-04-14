import random
import torch.utils.data
from pytracking import TensorDict
import numpy as np


def no_processing(data):
    return data


class SEMaskSampler(torch.utils.data.Dataset):
    """ similar to ATOMSampler """

    def __init__(self, datasets, p_datasets, samples_per_epoch, max_gap, num_test_frames=1, processing=no_processing,
                 frame_sample_mode='default'):
        """
        args:
            datasets - List of datasets to be used for training
            p_datasets - List containing the probabilities by which each dataset will be sampled
            samples_per_epoch - Number of training samples per epoch
            max_gap - Maximum gap, in frame numbers, between the train (reference) frame and the test frames.
            num_test_frames - Number of test frames used for calculating the IoU prediction loss.
            processing - An instance of Processing class which performs the necessary processing of the data.
            frame_sample_mode - Either 'default' or 'causal'. If 'causal', then the test frames are sampled in a causal
                                manner.
        """
        self.datasets = datasets

        # If p not provided, sample uniformly from all videos
        if p_datasets is None:
            p_datasets = [1 for d in self.datasets]

        # Normalize
        p_total = sum(p_datasets)
        self.p_datasets = [x/p_total for x in p_datasets]

        self.samples_per_epoch = samples_per_epoch
        self.max_gap = max_gap
        self.num_test_frames = num_test_frames
        self.num_train_frames = 1                         # Only a single train frame allowed
        self.processing = processing
        self.frame_sample_mode = frame_sample_mode

    def __len__(self):
        return self.samples_per_epoch

    def _sample_visible_ids(self, visible, num_ids=1, min_id=None, max_id=None):
        """ Samples num_ids frames between min_id and max_id for which target is visible

        args:
            visible - 1d Tensor indicating whether target is visible for each frame
            num_ids - number of frames to be samples
            min_id - Minimum allowed frame number
            max_id - Maximum allowed frame number

        returns:
            list - List of sampled frame numbers. None if not sufficient visible frames could be found.
        """
        if min_id is None or min_id < 0:
            min_id = 0
        if max_id is None or max_id > len(visible):
            max_id = len(visible)

        valid_ids = [i for i in range(min_id, max_id) if visible[i]]

        # No visible ids
        if len(valid_ids) == 0:
            return None

        return random.choices(valid_ids, k=num_ids)

    def __getitem__(self, index):
        """
        args:
            index (int): Index (Ignored since we sample randomly)

        returns:
            TensorDict - dict containing all the data blocks
        """
        dataset = random.choices(self.datasets, self.p_datasets)[0]
        is_video_dataset = dataset.is_video_sequence()

        min_visible_frames = 2 * (self.num_test_frames + self.num_train_frames)
        enough_visible_frames = False

        # Sample a sequence with enough visible frames and get anno for the same
        while not enough_visible_frames:
            seq_id = random.randint(0, dataset.get_num_sequences() - 1)
            seq_info_dict = dataset.get_sequence_info(seq_id)
            visible = seq_info_dict['visible']

            num_visible = visible.type(torch.int64).sum().item()
            # visible frames > 20 ==> ,visible frames > 20
            enough_visible_frames = ((not is_video_dataset) and num_visible > 0) or\
                                    (num_visible > min_visible_frames and (not dataset.has_mask()) and len(visible) >= 20) or \
                                    (num_visible > min_visible_frames and (dataset.has_mask()) and len(visible) >= 2)

        if is_video_dataset:
            train_frame_ids = None
            test_frame_ids = None
            gap_increase = 0
            if self.frame_sample_mode == 'default':
                # Sample frame numbers
                while test_frame_ids is None:
                    train_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_train_frames)
                    test_frame_ids = self._sample_visible_ids(visible, min_id=train_frame_ids[0] - self.max_gap - gap_increase,
                                                              max_id=train_frame_ids[0] + self.max_gap + gap_increase,
                                                              num_ids=self.num_test_frames)
                    gap_increase += 5   # Increase gap until a frame is found
            elif self.frame_sample_mode == 'causal':
                # Sample frame numbers in a causal manner, i.e. test_frame_ids > train_frame_ids
                while test_frame_ids is None:
                    base_frame_id = self._sample_visible_ids(visible, num_ids=1, min_id=self.num_train_frames - 1,
                                                             max_id=len(visible)-self.num_test_frames)
                    prev_frame_ids = self._sample_visible_ids(visible, num_ids=self.num_train_frames - 1,
                                                              min_id=base_frame_id[0] - self.max_gap - gap_increase,
                                                              max_id=base_frame_id[0])
                    if prev_frame_ids is None:
                        gap_increase += 5
                        continue
                    train_frame_ids = base_frame_id + prev_frame_ids
                    test_frame_ids = self._sample_visible_ids(visible, min_id=train_frame_ids[0]+1,
                                                              max_id=train_frame_ids[0] + self.max_gap + gap_increase,
                                                              num_ids=self.num_test_frames)
                    gap_increase += 5   # Increase gap until a frame is found
            else:
                raise ValueError('Unknown frame_sample_mode.')
        else:
            train_frame_ids = [1]*self.num_train_frames
            test_frame_ids = [1]*self.num_test_frames

        # Get frames
        if not dataset.has_mask():
            # standard procedure as ATOM
            train_frames, train_anno_dict, _ = dataset.get_frames(seq_id, train_frame_ids, seq_info_dict)
            train_anno = train_anno_dict['bbox']

            test_frames, test_anno_dict, _ = dataset.get_frames(seq_id, test_frame_ids, seq_info_dict)
            test_anno = test_anno_dict['bbox']

            # Prepare data
            H,W,_ = train_frames[0].shape
            data = TensorDict({'train_images': train_frames,
                               'train_masks': [np.zeros((H,W,1))],
                               'train_anno': train_anno, #list [(4,) torch tensor]
                               'test_images': test_frames,
                               'test_masks': [np.zeros((H,W,1))],
                               'test_anno': test_anno, # list [(4,) torch tensor]
                               'dataset': dataset.get_name(),
                               'mask':False})

        else:
            # if mask exists in data, process it here
            train_frames, train_masks, train_anno_dict, _ = dataset.get_frames(seq_id, train_frame_ids, seq_info_dict)
            train_anno = train_anno_dict['bbox']
            test_frames, test_masks, test_anno_dict, _ = dataset.get_frames(seq_id, test_frame_ids, seq_info_dict)
            test_anno = test_anno_dict['bbox']
            # Prepare data
            data = TensorDict({'train_images': train_frames,
                               'train_masks': train_masks, # [ndarray (H,W,1)]
                               'train_anno': train_anno,  # list [(4,) torch tensor]
                               'test_images': test_frames,
                               'test_masks': test_masks, # [ndarray (H,W,1)]
                               'test_anno': test_anno,  # list [(4,) torch tensor]
                               'dataset': dataset.get_name(),
                               'mask':True})
        # Send for processing
        data = self.processing(data)
        return data
