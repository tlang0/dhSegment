import io
import os
import shutil
import subprocess
import tempfile
from glob import glob

import pandas as pd
from scipy.misc import imread, imresize, imsave
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir, os.path.pardir)))
from exps.post_processing import PAGE
from doc_seg.utils import load_pickle

CBAD_JAR = '/home/datasets/TranskribusBaseLineEvaluationScheme_v0.1.3/' \
           'TranskribusBaseLineEvaluationScheme-0.1.3-jar-with-dependencies.jar'
COPIED_GT_PAGE_DIR_NAME = 'gt_page'


def cbad_evaluate_folder(output_folder: str, validation_dir: str, verbose=False,
                        debug_folder=None, jar_path: str = CBAD_JAR) -> dict:
    """

    :param output_folder: contains the *.pkl output files of the post-processing
    :param validation_dir: contains 'images' 'labels' and other folders
    :param jar_path:
    :return:
    """

    if debug_folder is not None:
        os.makedirs(debug_folder, exist_ok=True)

    # Copy xml gt PAGE to output directory in order to use java evaluation tool
    with tempfile.TemporaryDirectory() as tmpdirname:
        gt_dir = os.path.join(validation_dir, 'gt')

        filenames_processed = glob(os.path.join(output_folder, '*.pkl'))
        filenames_processed.extend(glob(os.path.join(output_folder, '*.xml')))

        xml_filenames_list = list()
        for filename in filenames_processed:
            basename = os.path.basename(filename).split('.')[0]
            gt_page = PAGE.parse_file(os.path.join(gt_dir, '{}.xml'.format(basename)))
            xml_filename = os.path.join(tmpdirname, basename + '.xml')
            if filename[-4:] == '.pkl':
                contours, img_shape = load_pickle(filename)
            else:
                extracted_page = PAGE.parse_file(filename)
                img_shape = (extracted_page.image_height, extracted_page.image_width)
                contours = [PAGE.Point.list_to_cv2poly(tl.baseline)
                            for tr in extracted_page.text_regions for tl in tr.text_lines]
            ratio = (gt_page.image_height/img_shape[0], gt_page.image_width/img_shape[1])
            PAGE.save_baselines(xml_filename, contours, ratio, initial_shape=img_shape[:2])

            gt_xml_file = os.path.join(gt_dir, basename + '.xml')
            xml_filenames_list.append((gt_xml_file, xml_filename))

            if debug_folder is not None:
                img = imread(os.path.join(validation_dir, 'images', basename+'.jpg'))
                img = imresize(img, 1000/img.shape[0])
                gt_page.draw_baselines(img, color=(0, 255, 0))
                generated_page = PAGE.parse_file(xml_filename)
                generated_page.draw_baselines(img, color=(255, 0, 0))
                imsave(os.path.join(debug_folder, basename+'.jpg'), img)

        gt_pages_list_filename = os.path.join(tmpdirname, 'gt_pages.lst')
        generated_pages_list_filename = os.path.join(tmpdirname, 'generated_pages.lst')
        with open(gt_pages_list_filename, 'w') as f:
            f.writelines('\n'.join([s[0] for s in xml_filenames_list]))
        with open(generated_pages_list_filename, 'w') as f:
            f.writelines('\n'.join([s[1] for s in xml_filenames_list]))

        # Run command line evaluation tool
        cwd = os.getcwd()
        os.chdir(os.path.dirname(gt_pages_list_filename))
        cmd = 'java -jar {} {} {}'.format(jar_path, gt_pages_list_filename, generated_pages_list_filename)
        result = subprocess.check_output(cmd, shell=True).decode()
        if debug_folder is not None:
            with open(os.path.join(debug_folder, 'scores.txt'), 'w') as f:
                f.write(result)
            parse_score_txt(result, os.path.join(debug_folder, 'scores.csv'))

        lines = result.splitlines()
        avg_precision = float(next(filter(lambda l: 'Avg (over pages) P value:' in l, lines)).split()[-1])
        avg_recall = float(next(filter(lambda l: 'Avg (over pages) R value:' in l, lines)).split()[-1])
        f_measure = float(next(filter(lambda l: 'Resulting F_1 value:' in l, lines)).split()[-1])
        os.chdir(cwd)
        return {
            'avg_precision': avg_precision,
            'avg_recall': avg_recall,
            'f_measure': f_measure
        }


def parse_score_txt(score_txt, output_csv):
    lines = score_txt.splitlines()
    header_ind = next((i for i, l in enumerate(lines)
                       if l == '#P value, #R value, #F_1 value, #TruthFileName, #HypoFileName'))
    final_line = next((i for i, l in enumerate(lines) if i > header_ind and l == ''))
    csv_data = '\n'.join(lines[header_ind:final_line])
    df = pd.read_csv(io.StringIO(csv_data))
    df = df.rename(columns={k: k.strip() for k in df.columns})
    df['#HypoFileName'] = [os.path.basename(f).split('.')[0] for f in df['#HypoFileName']]
    del df['#TruthFileName']
    df = df.rename(columns={'#P value': 'P', '#R value': 'R', '#F_1 value': 'F_1', '#HypoFileName': 'basename'})
    df = df.reindex(columns=['basename', 'F_1', 'P', 'R'])
    df = df.sort_values('F_1', ascending=True)
    df.to_csv(output_csv, index=False)


def get_gt_page_filename(exported_xml):
    return os.path.join(os.path.dirname(exported_xml), COPIED_GT_PAGE_DIR_NAME,
                        '{}.xml'.format(os.path.basename(exported_xml).split('.')[0]))


def copy_gt_page_to_exp_directory(target_page_dir: str, gt_dir: str) -> None:
    """
    Copies the original gt PAGE xml file to the postprocessing directory to be able to launch the
    java tool for evaluation
    :param target_page_dir: postprocessing dir
    :param gt_dir: original PAGE directories (usually organized this way : gt_dir/<collection_id>/page/*.xml)
    :return:
    """
    os.makedirs(target_page_dir, exist_ok=True)
    if len(glob(os.path.join(target_page_dir, '*'))) > 0:
        print('Already copied PAGE files.')
        return

    xml_filenames_list = glob(os.path.join(gt_dir, '**', 'page', '*.xml'))

    for filename in xml_filenames_list:
        directory, basename = os.path.split(filename)
        acronym = directory.split(os.path.sep)[-2].split('_')[0]
        new_filenname = '{}_{}.xml'.format(acronym, basename.split('.')[0])
        shutil.copy(filename, os.path.join(target_page_dir, new_filenname))
