#!/usr/bin/env python

import numpy as np
import sys
from imageio import imread

BGLABEL = 0

def eval_jaccard(filepath_result, filepath_gt):
    img_result = imread(filepath_result)
    img_gt = imread(filepath_gt)

    maxlabel = np.amax(img_gt)
    results = []
    intersection_area_overall = 0
    unions_overall = np.logical_or(np.not_equal(img_result, BGLABEL), np.not_equal(img_gt, BGLABEL))
    for label in range(maxlabel + 1):
        labelimg_gt = np.equal(label, img_gt)
        labelimg_result = np.equal(label, img_result)
        intersection = np.logical_and(labelimg_result, labelimg_gt)
        intersection_area = np.count_nonzero(intersection)
        #union = np.logical_or(labelimg_result, labelimg_gt)
        #count_union = np.count_nonzero(union)
        union_area = np.count_nonzero(labelimg_gt) + np.count_nonzero(labelimg_result) - intersection_area
        jaccard = intersection_area / union_area
        # add tuple
        results.append((label, jaccard, intersection_area, union_area))
        if label != BGLABEL: # ignore background for overall values
            intersection_area_overall += intersection_area

    union_area_overall = np.count_nonzero(unions_overall)
    jaccard_overall = intersection_area_overall / union_area_overall

    return ((jaccard_overall, intersection_area_overall, union_area_overall), results)

def main():
    if len(sys.argv) == 3:
        result_overall, results = eval_jaccard(sys.argv[1], sys.argv[2])
        if len(results) == 0:
            print("no results")
        else:
            for (label, jaccard, intersection_area, union_area) in results:
                backgroundtext = " (bg)" if label == BGLABEL else ""
                print("class {}{}: jaccard = {}, ({}/{})".format(label, backgroundtext, jaccard, intersection_area, union_area))

            jaccard_overall, intersection_area_overall, union_area_overall = result_overall
            print("overall: jaccard = {}, ({}/{})".format(jaccard_overall, intersection_area_overall, union_area_overall))
    else:
        print("usage:\n{} <filepath_result> <filepath_gt>".format(sys.argv[0]))

if __name__ == '__main__':
    main()