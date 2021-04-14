import os
from got10k.trackers import Tracker
from got10k.experiments import ExperimentOTB, ExperimentUAV123, ExperimentNfS, ExperimentTColor128
from arena.GOT10k.common_path import _dataset_name, _dataset_root, save_dir

from arena.GOT10k import DiMPsuper_RF, DiMP50_RF, ATOM_RF, ECO_RF, SiamRPNpp_RF, RTMDNet_RF

Trackers = {
    'DiMPsuper_RF': DiMPsuper_RF,
    'DiMP50_RF': DiMP50_RF,
    'ATOM_RF': ATOM_RF,
    'ECO_RF': ECO_RF,
    'SiamRPNpp_RF': SiamRPNpp_RF,
    'RTMDNet_RF': RTMDNet_RF
}



def check_dir(dir):
    if not os.path.exists(dir):
        print('creating directory: {}'.format(dir))
        os.makedirs(dir)
    print('using directory: {}'.format(dir))
    return dir


class IdentityTracker(Tracker):
    def __init__(self):
        super(IdentityTracker, self).__init__(name='IdentityTracker')

    def init(self, image, box):
        self.box = box

    def update(self, image):
        return self.box


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracker', default='DiMPsuper_RF',
                        help='whether visualzie result')
    parser.add_argument('--rf_code', default='a',
                        help='whether visualzie result')
    parser.add_argument('--overwrite', action='store_true',
                        help='whether overwrite results')
    parser.add_argument('--vis', action='store_true', default=False,
                        help='whether visualzie result')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # setup tracker
    tracker = Trackers[args.tracker](args.rf_code, False)
    exps = [(ExperimentOTB, os.path.join(_dataset_root, 'OTB100')),
            (ExperimentUAV123, os.path.join(_dataset_root, 'UAV123_fix')),
            (ExperimentNfS, os.path.join(_dataset_root, 'nfs')),
            (ExperimentTColor128, os.path.join(_dataset_root, 'Temple-color-128'))]

    for Experiment, data_root in exps:
        _save_dir = os.path.join(save_dir, os.path.basename(data_root))
        # run experiments on GOT-10k (validation subset)
        experiment = Experiment(
            data_root,
            result_dir=check_dir(os.path.join(_save_dir, 'results')),
            report_dir=check_dir(os.path.join(_save_dir, 'reports'))
        )
        experiment.run(tracker, visualize=args.vis)

        # report performance
        experiment.report([tracker.name])
