#!/usr/bin/env python

import numpy as np
import sys
from imageio import imread, imsave

def eval_jaccard(filepath_result, filepath_gt):
    img_result = imread(filepath_result)
    img_gt = imread(filepath_gt)

    maxlabel = np.amax(img_gt)
    results = []
    for label in range(maxlabel + 1):
        labelimg_gt = np.equal(label, img_gt)
        labelimg_result = np.equal(label, img_result)
        intersection = np.logical_and(labelimg_result, labelimg_gt)
        union = np.logical_or(labelimg_result, labelimg_gt)
        count_intersection = np.count_nonzero(intersection)
        count_union = np.count_nonzero(union)
        jaccard = count_intersection / count_union
        results.append((label, jaccard, count_intersection, count_union))

    return results

def main():
    if len(sys.argv) == 3:
        results = eval_jaccard(sys.argv[1], sys.argv[2])
        if len(results) == 0:
            print("no results")
        else:
            for (label, jaccard, count_intersection, count_union) in results:
                print("class {}: jaccard = {}, ({}/{})".format(label, jaccard, count_intersection, count_union))
    else:
        print("usage:\n{} <filepath_result> <filepath_gt>".format(sys.argv[0]))

if __name__ == '__main__':
    main()