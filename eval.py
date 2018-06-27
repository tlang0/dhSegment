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
    """results_batch must be a list of tuples (imagename, data), where data is a list of tuples"""
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

    result_path_str = sys.argv[1]
    gt_path_str = sys.argv[2]
    output_path_str = sys.argv[3]

    result_path = Path(result_path_str)
    gt_path = Path(gt_path_str)

    if result_path.is_file() and gt_path.is_file(): # single image
        print("single image...")
        imagename = result_path.stem
        jaccard_overall, data = eval_jaccard(result_path_str, gt_path_str)
        if len(data) == 0:
            print("no results")
            return
        write_results(output_path_str, [(imagename, data)])
        #print(data)
        print("overall jaccard = {}".format(jaccard_overall))
    elif result_path.is_dir() and gt_path.is_dir(): # batch
        print("batch...")
        results_batch = []
        for file_gt in gt_path.iterdir():
            imagename = file_gt.stem
            files_matching = list(result_path.glob("*{}*.*".format(imagename)))
            if len(files_matching) == 0:
                print("no matching file was found for image: {}".format(imagename))
                continue
            if len(files_matching) > 1:
                print("warning: more than one matching file was found for image: {}".format(imagename))

            #print(str(file_gt), str(files_matching[0]))
            jaccard_overall, data = eval_jaccard(str(files_matching[0]), str(file_gt))
            if len(data) == 0:
                print("no results for image {}".format(imagename))
                continue
            print("image {}: overall jaccard = {}".format(imagename, jaccard_overall))
            results_batch.append((imagename, data))
        write_results(output_path_str, results_batch)
        print("finished. evaluation results of {} images have been stored.".format(len(results_batch)))
    else:
        print("input paths must either both be files or directories!")

if __name__ == '__main__':
    main()