#!/usr/bin/env python2
# -*- coding: utf-8 -*-
# $File: detection_eval.py
# $Date: Fri Mar 20 11:11:42 2015 +0800
# $Author: Xinyu Zhou <zxytim[at]gmail[dot]com>

import numpy as np
import cv2
from collections import defaultdict, namedtuple
from itertools import count
from bisect import bisect
import argparse


Circle = namedtuple('Circle', 'center, radius')


def find_best_circle(poly):
    """ poly: a list of 4 points
        return a circle in ((x, y), r) , who has a largest IOU with poly
    """
    def get_poly_points(poly):
        PADDING = 10
        FILL = 255
        poly = np.asarray(poly, dtype=np.int32)
        maxx, maxy = max([k[0] for k in poly]), max([k[1] for k in poly])
        minx, miny = min([k[0] for k in poly]), min([k[1] for k in poly])
        img = np.zeros((maxy + PADDING, maxx + PADDING), dtype='uint8')
        cv2.fillConvexPoly(img, poly, FILL)

        points = []
        for x in xrange(minx, maxx + 1):
            for y in xrange(miny, maxy + 1):
                if img[y, x] == FILL:
                    points.append((x, y))
        return np.asarray(points)

    def get_com(points):
        return np.round(np.mean(points, axis=0))

    def find_best_radius(point, points):
        def dist(p):
            return (point[0] - p[0]) ** 2 + (point[1] - p[1]) ** 2
        dists = sorted([dist(p) for p in points])
        maxr = int(np.sqrt(dists[-1])) + 1
        best_iou = 0.0
        best_r = 0
        for r in xrange(maxr):
            idx = bisect(dists, r * r)  # i
            circle_area = np.pi * r * r
            union = len(points) + circle_area - idx
            iou = float(idx) / union
            if iou > best_iou:
                best_iou = iou
                best_r = r
        return best_r, best_iou

    points = get_poly_points(poly)

    best_p, best_r, best_iou = None, None, 0.0
    best_p = get_com(points)
    best_r, best_iou = find_best_radius(best_p, points)
    #for p in points:
        #r, iou = find_best_radius(p, points)
        #if iou > best_iou:
            #best_iou = iou
            #best_p = p
            #best_r = r
    return Circle(best_p, best_r)
#     return best_p, best_r, best_iou


TextBox = namedtuple('TextBox', 'tl_id, poly, text')


def read_text_box_gt(path):
    ret = []
    with open(path) as f:
        for line in f:
            tl_id, poly, text = line.rstrip().decode(
                'utf-8').split('\t')
            tl_id = int(tl_id)
            poly = np.asarray(map(int, poly.split())).reshape(-1, 2)
            ret.append(TextBox(tl_id, poly, text))
    return ret


def read_prediction_poly(path):
    with open(path) as f:
        return np.array([map(float, line.rstrip().split())
                 for line in f], dtype='float32').reshape(-1, 4, 2)


def do_read_prediction_circle(path):
    with open(path) as f:
        lines = [map(float, line.rstrip().split())
                 for line in f]
    return [Circle((x, y), r) for x, y, r in lines]


def eval_circles(img_shape, gt, pred):
    gt_img = np.zeros(img_shape, dtype='uint8')
    pred_img = np.zeros(img_shape, dtype='uint8')

    for c in gt:
        cv2.circle(gt_img, tuple(map(int, c.center)), int(c.radius), 255,
                   cv2.cv.CV_FILLED)
    for c in pred:
        cv2.circle(pred_img, tuple(map(int, c.center)), int(c.radius), 255,
                   cv2.cv.CV_FILLED)

    gt_cnt = (gt_img > 0).sum()
    pred_cnt = (pred_img > 0).sum()
    corr = ((gt_img > 0) * (pred_img > 0)).sum()

    def score(tot):
        if tot == 0:
            return 1.0
        else:
            return corr / float(tot)

    recall = score(gt_cnt)
    precision = score(pred_cnt)
    return precision, recall


def read_prediction_circle(path, format):
    if format == 'text_box':
        pred_polys = read_prediction_poly(path)
        pred_circles = map(find_best_circle, pred_polys)
    else:
        assert format == 'circle'
        pred_circles = do_read_prediction_circle(path)
    return pred_circles


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(dest='image_path')
    parser.add_argument(dest='predictions',
                        help='lines of text boxes(8 numbers a line) if format '
                        'is set to text_box; lines of circles (3 numbers, '
                        '2 for center coordinate and one for radius) '
                        'if format is circle')
    parser.add_argument('--format', default='circle',
                        choices=['text_box', 'circle'])

    args = parser.parse_args()

    text_box_gt_path = args.image_path + '.text_box'
    img = cv2.imread(args.image_path)
    shape = img.shape[:2]

    gt_text_boxes = read_text_box_gt(text_box_gt_path)
    gt_polys = [tb.poly for tb in gt_text_boxes]
    gt_circles = map(find_best_circle, gt_polys)


    pred_circles = read_prediction_circle(args.predictions, args.format)

    precision, recall = eval_circles(shape, gt_circles, pred_circles)
    print precision, recall


if __name__ == '__main__':
    main()
# vim: foldmethod=marker
