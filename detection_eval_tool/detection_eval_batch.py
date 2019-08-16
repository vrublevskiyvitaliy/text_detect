#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: detection_eval_batch.py
# $Date: Fri Mar 20 15:10:27 2015 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>

import sys

sys.path.append('/usr/local/lib/python2.7/site-packages')

import argparse
import os
import cv2
import numpy as np
import imghdr
from collections import namedtuple

from detection_eval import read_text_box_gt, read_prediction_circle, \
    find_best_circle, eval_circles

TextLine = namedtuple('TextLine', 'translucent, english_only, poly, text')


NR_TYPES = 9
TYPE_NAMES = [
    'translucent & not english only',
    'not translucent & english only',
    'translucent & not english only',
    'translucent & english only',
    'translucent',
    'not translucent',
    'english only',
    'not english only',
    'all',
]


TYPE_HELP = '''
available types are:
{}
'''.format('\n'.join(
    '{}. {};'.format(i, TYPE_NAMES[i]) for i in range(NR_TYPES)
))


EPILOG = '''
IMPORTANT:
    1. Files in prediction_directory must have a `.pred' suffix, e.g., if
        there's an image `0.png' in ground_truth_directory, your corresponding
        output should have the name `0.png.pred' in prediction_directory
    2. Two input formats are available: `text_box' and `circle'. Where
        in `text_box' format, each line of prediction file contains 8 numbers,
        denoting coordinates of individual text_box; in `circle' format, each
        line contains only 3 numbers, stands for the coordinate of the center
        of the circle (2 numbers, x, y)
'''


def read_text_line_gt(path):
    ret = []
    with open(path) as f:
        for line in f:
            translucent, english_only, poly, text = line.rstrip().decode(
                'utf-8').split('\t')
            english_only = int(english_only)
            translucent = int(translucent)
            poly = np.asarray(map(int, poly.split())).reshape(-1, 2)
            ret.append(TextLine(translucent, english_only, poly, text))
    return ret


class Stat(object):
    def __init__(self):
        self.precision_sum = 0.0
        self.recall_sum = 0.0
        self.count = 0

    def add(self, precision, recall):
        self.count += 1
        self.precision_sum += precision
        self.recall_sum += recall

    @property
    def precision(self):
        if self.count == 0:
            return 1.0
        return self.precision_sum / self.count

    @property
    def recall(self):
        if self.count == 0:
            return 1.0
        return self.recall_sum / self.count

    @property
    def f1(self):
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r)


def main():
    parser = argparse.ArgumentParser(epilog=EPILOG)
    parser.add_argument(dest='ground_truth_directory')
    parser.add_argument(dest='prediction_directory')
    parser.add_argument('--type', default=8, type=int,
                        choices=range(NR_TYPES),
                        help=TYPE_HELP)
    parser.add_argument('--format', default='circle',
                        choices=['text_box', 'circle'],
                        help='default=circle')
    args = parser.parse_args()

    assert os.path.isdir(args.ground_truth_directory)
    assert os.path.isdir(args.prediction_directory)

    stat = Stat()

    for dirpath, dirnames, filenames in os.walk(args.ground_truth_directory):
        for fname in filenames:
            fpath = os.path.join(dirpath, fname)
            if imghdr.what(fpath) is not None:
                img = cv2.imread(fpath)
                if img is None:
                    continue
                shape = img.shape[:2]
                pred_path = os.path.join(
                    args.prediction_directory, fname + '.pred')
                if not os.path.exists(pred_path):
                    continue

                text_boxes = read_text_box_gt(fpath + '.text_box')
                text_lines = read_text_line_gt(fpath + '.gt')

                pred_circles = read_prediction_circle(pred_path, args.format)

                def tb_is_type(tb, typ):
                    ''':param typ: if typ < 4, it is the mask of whether
                    translucent and english_only;
                    typ in {4,5} means translucent or not, and don't care if it
                    is english_only;
                    typ == {6,7} means english_only or not, and don't care if
                    it is translucent;
                    typ == 8 means all text boxes are counted'''
                    tl = text_lines[tb.tl_id]
                    tr = tl.translucent
                    en = tl.english_only
                    if typ < 4:
                        return tr * 2 + en == typ
                    if typ in {4, 5}:
                        return (5 - typ) == tr
                    if typ in {6, 7}:
                        return (7 - typ) == en
                    assert typ == 8, typ
                    return True

                gt_circles = [find_best_circle(tb.poly) for tb in text_boxes
                              if tb_is_type(tb, args.type)]
                stat.add(*eval_circles(shape, gt_circles, pred_circles))

    print stat.precision, stat.recall, stat.f1


if __name__ == '__main__':
    main()

# vim: foldmethod=marker
