def param_setup(settings):
    """ Configure the training parameters for AlphaRefine """

    # Most common settings are assigned in the settings struct
    settings.description = 'Settings of SEcm module'

    ''' !!! some important hyperparameters !!! '''
    settings.learning_rate = 1e-3  # Batch size
    settings.batch_size = 32  # Batch size
    settings.search_area_factor = 2.0  # Image patch size relative to target size
    settings.feature_sz = 16  # Size of feature map
    settings.output_sz = settings.feature_sz * 16  # Size of input image patches
    settings.used_layers = ['layer3']

    # Settings for the image sample and proposal generation
    settings.center_jitter_factor = {'train': 0, 'test': 0.25}
    settings.scale_jitter_factor = {'train': 0, 'test': 0.25}
    settings.max_gap = 50
    settings.sample_per_epoch = 4000
    settings.save_interval = 5  # the interval of saving the checkpoints

    # others
    settings.print_interval = 100  # How often to print loss and other info
    settings.num_workers = 4  # Number of workers for image loading
    settings.normalize_mean = [0.485, 0.456, 0.406]  # Normalize mean (default pytorch ImageNet values)
    settings.normalize_std = [0.229, 0.224, 0.225]  # Normalize std (default pytorch ImageNet values)
