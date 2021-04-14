import os
import json
from arena.GOT10k.common_path import save_dir


def parse_args():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--report_dir', default=save_dir, help='path to the report directory')
    parser.add_argument('--dataset', default='', help='the dataset to be evaluated')
    parser.add_argument('--prefix', default='', help='the dataset to be evaluated')
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    report_dir = os.path.join(args.report_dir, args.dataset, 'reports')
    report_dir = os.path.join(report_dir, os.listdir(report_dir)[0])
    items = sorted([os.path.join(report_dir, it) for it in os.listdir(report_dir)])
    for item in items:
        exp_name = os.path.basename(item); print(exp_name)
        json_file = os.path.join(item, 'performance.json')
        with open(json_file) as fid:
            report = json.load(fid)
        report_content = list(report.items())[0][1]
        overall_performance = report_content['overall']
        print("AUC:{} \nPrecision:{}\n\n".format(overall_performance['success_score'], overall_performance['precision_score']))
