#!/usr/bin/env python

import numpy as np
import sys
from imageio import imread
import skimage.measure
from pathlib import Path, PurePath

BGLABEL = 0

def eval_jaccard(filepath_result, filepath_gt):
    img_result = imread(filepath_result)
    img_gt = imread(filepath_gt)

    maxlabel = np.amax(img_gt)
    intersection_area_overall = 0
    unions_overall = np.logical_or(np.not_equal(img_result, BGLABEL), np.not_equal(img_gt, BGLABEL))
    data = []
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
        data.append((label, num_gt_regions, jaccard))

    data_mat = np.asarray(data, dtype=np.double)
    # mean jaccard index weighted by number of components per class
    jaccard_overall = np.dot(data_mat[:, 1], data_mat[:, 2]) / np.sum(data_mat[:, 1])

    return jaccard_overall, data

def write_results(output_path, results_batch):
    with open(output_path, "w") as f:
        # iterate over results of all images
        for results in results_batch:
            imagename, data = results
            for tpl in data:
                f.write(';'.join(map(str, (imagename,) + tpl)) + '\n')

def main():
    if len(sys.argv) != 4:
        print("usage:\n{} <filepath_result> <filepath_gt> <output_path>".format(sys.argv[0]))
        return

    result_path = sys.argv[1]
    gt_path = sys.argv[2]
    output_path = sys.argv[3]

    #if Path(result_path).is_file() and Path(gt_path).is_file():
    imagename = PurePath(result_path).stem

    jaccard_overall, data = eval_jaccard(result_path, gt_path)
    if len(data) == 0:
        print("no results")
        return

    write_results(output_path, [(imagename, data)])

    print(jaccard_overall)
    print(data)

if __name__ == '__main__':
    main()