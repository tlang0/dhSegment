#!/usr/bin/env python

import numpy as np
import sys
from imageio import imread
import skimage.measure

BGLABEL = 0

def eval_jaccard(filepath_result, filepath_gt):
    img_result = imread(filepath_result)
    img_gt = imread(filepath_gt)

    maxlabel = np.amax(img_gt)
    intersection_area_overall = 0
    unions_overall = np.logical_or(np.not_equal(img_result, BGLABEL), np.not_equal(img_gt, BGLABEL))
    results = []
    for label in range(maxlabel + 1):
        if label == BGLABEL:
            continue
        labelimg_gt = np.equal(label, img_gt)
        gt_area = np.count_nonzero(labelimg_gt)
        if gt_area <= 0:
            continue
        labelimg_result = np.equal(label, img_result)
        intersection = np.logical_and(labelimg_result, labelimg_gt)
        intersection_area = np.count_nonzero(intersection)
        union_area = gt_area + np.count_nonzero(labelimg_result) - intersection_area
        jaccard = intersection_area / union_area
        # count connected components in gt image
        _, num_gt_regions = skimage.measure.label(labelimg_gt, return_num=True)
        results.append((label, num_gt_regions, jaccard))

    results_mat = np.asarray(results, dtype=np.double)
    # mean jaccard index weighted by number of components per class
    jaccard_overall = np.dot(results_mat[:, 1], results_mat[:, 2]) / np.sum(results_mat[:, 1])

    return jaccard_overall, results_mat

def main():
    if len(sys.argv) == 3:
        jaccard_overall, results_mat = eval_jaccard(sys.argv[1], sys.argv[2])
        if len(results_mat) == 0:
            print("no results")
        else:
            print(jaccard_overall)
            print(results_mat)
    else:
        print("usage:\n{} <filepath_result> <filepath_gt>".format(sys.argv[0]))

if __name__ == '__main__':
    main()