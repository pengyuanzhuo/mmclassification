import os
import os.path as osp
import argparse
from sklearn import metrics
from sklearn.metrics import confusion_matrix
import numpy as np


def parse():
    parser = argparse.ArgumentParser('eval f1')
    parser.add_argument('gt')
    parser.add_argument('pred')
    return parser.parse_args()


def load_ann(file):
    label_dict = {}
    with open(file, 'r') as f:
        for line in f:
            imgname, label = line.strip().split(',')
            if imgname == 'image_name':
                continue
            label_dict[imgname] = int(label)

    return label_dict


def evaluation(confusion_matrix):
    confusion_matrix = confusion_matrix.astype(np.float32)
    metrics = {}
    num_classes = confusion_matrix.shape[0]

    precisions = []
    recalls = []
    f1s = []
    for i in range(num_classes):
        p = confusion_matrix[i][i] / confusion_matrix.sum(axis=0)[i]
        r = confusion_matrix[i][i] / confusion_matrix.sum(axis=1)[i]
        f1 = 2 * p * r / (p + r)

        precisions.append(p)
        recalls.append(r)
        f1s.append(f1)

    accuracy = confusion_matrix.trace() / confusion_matrix.sum()
    metrics['accuracy'] = accuracy
    metrics['precisions'] = precisions
    metrics['recalls'] = recalls
    metrics['f1'] = f1s
    metrics['avg_f1'] = sum(f1s) / num_classes
 
    return metrics


def print_metrics(metrics_dict):
    print("#" * 50)
    print('ACC: ', metrics['accuracy'])
    print('AVG F1: ', metrics['avg_f1'])
    for i in range(len(metrics['f1'])):
        print('CLASS {} F1: {}'.format(i, metrics['f1'][i]))
        print('CLASS {} P: {}'.format(i, metrics['precisions'][i]))
        print('CLASS {} R: {}'.format(i, metrics['recalls'][i]))
    print("#" * 50)


if __name__ == '__main__':
    args = parse()

    gt_dict = load_ann(args.gt)
    pred_dict = load_ann(args.pred)
    assert len(gt_dict) == len(pred_dict)

    gt_list = []
    pred_list = []
    for imgname, gt_label in gt_dict.items():
        gt_list.append(gt_label)
        pred_list.append(pred_dict[imgname])

    cm = confusion_matrix(gt_list, pred_list)
    print('confusion matrix: \n', cm)

    metrics = evaluation(cm)
    print_metrics(metrics)



