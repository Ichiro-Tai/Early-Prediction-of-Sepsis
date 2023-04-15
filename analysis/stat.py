import os
import sys
sys.path.append('../')

from tools import parse, python_utils
import numpy as np

args = parse.args

analysis_path = os.path.join(args.result_dir, 'feature_label_count.npy')

def ana_feat_dist(task):
    n_split = 100
    feature_label_count = np.zeros((143, 2, n_split))
    patient_time_record_dict = python_utils.myreadjson(os.path.join(args.result_dir, 'patient_time_record_dict.json'.format(args.task)))
    patient_label_dict = python_utils.myreadjson(os.path.join(args.result_dir, 'patient_label_dict.json'))
    [ [ [0. for _ in range(n_split)], [0. for _ in range(n_split)] ] for i in range(143) ]
    for ip, (p, t_dict) in enumerate(patient_time_record_dict.items()):
        if ip % 10000 == 0:
            print(ip, len(patient_time_record_dict))

        label = patient_label_dict[p]
        for t, vs in list(t_dict.items()):
            for v in vs:
                feature, value = v
                idx = int(value * n_split)
                feature_label_count[feature, label, idx] += 1
    for f in range(143):
        for l in range(2):
            feature_label_count[feature, label] /= feature_label_count[feature, label].sum()
    np.save(analysis_path, feature_label_count)


def draw_pic():
    def avg(ys, n = 50):
        nys = []
        for i,y in enumerate(ys):
            st = max(0, i - n)
            en = min(len(ys), i + n + 1) 
            nys.append(np.mean(ys[st:en]))

        return nys

    import matplotlib.pyplot as plt
    flc = np.load(analysis_path)
    for f in range(143):
        lc = flc[f]
        x = list(range(len(lc[0])))
        plt.plot(x,avg(lc[0]),'b')
        plt.plot(x,avg(lc[1]),'r')
        plt.savefig(os.path.join(args.result_dir, 'fig/{:d}.png'.format(f)))
        plt.clf()
        if f > 10:
            break

def main():
    ana_feat_dist('task1')
    draw_pic()


if __name__ == '__main__':
    os.system('clear')
    main()
