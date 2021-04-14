import os
from got10k.trackers import Tracker
from got10k.experiments import ExperimentGOT10k
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

save_dir = os.path.join(save_dir, _dataset_name)


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
    tracker = Trackers[args.tracker](args.rf_code)

    # run experiments on GOT-10k (validation subset)
    data_root = os.path.join(_dataset_root, 'GOT10k')
    experiment = ExperimentGOT10k(
        data_root, subset='test',
        result_dir=check_dir(os.path.join(save_dir, 'results')),
        report_dir=check_dir(os.path.join(save_dir, 'reports'))
    )
    experiment.run(tracker, visualize=args.vis, overwrite_result=args.overwrite)

    # report performance
    experiment.report([tracker.name])
