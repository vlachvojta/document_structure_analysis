#!/usr/bin/python3
"""Load label studio JSON results and cut out objects from images.

Name them as <image_name>_<object_label>_<object_id>.<image_extension>.

Usage:
$ python3 cut_annotations.py -i <image_folder> -l <label_file> -o <output_folder>
Resulting in <output_folder> with cut out objects, named as <image_name>_<object_label>_<object_id>.<image_extension>.
"""

import argparse
import re
import sys
import os
import time
import logging
import cv2
from tqdm import tqdm

# add parent directory to python file path to enable imports
file_dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_dirname)
sys.path.append(os.path.dirname(file_dirname))

from dataset.label_studio_utils import LabelStudioResults, label_studio_coords_to_xywh, add_padding


def parseargs():
    """Parse arguments."""
    print('')
    print('sys.argv: ')
    print(' '.join(sys.argv))
    print('--------------------------------------')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--image-folder", default='example_data/0_images',
        help="Input folder where to look for images.")
    parser.add_argument(
        "-l", "--label-file", type=str, default='example_data/0_annotated_table_detection.json',
        help="Label studio JSON export file.")
    parser.add_argument(
        '-f', '--filter-labels', nargs='+', default=[],
        help="Filter labels to keep. Default: all labels.")
    parser.add_argument(
        '-p', '--padding', type=int, default=0,
        help="Padding around the object.")
    parser.add_argument(
        "-o", "--output-folder", type=str, default='example_data/1_cut',
        help="Output folder where to save cut out objects.")
    parser.add_argument(
        '-v', "--verbose", action='store_true', default=False,
        help="Activate verbose logging.")

    return parser.parse_args()


def main():
    """Main function for simple testing"""
    args = parseargs()

    start = time.time()

    cutter = AnotationCutter(
        image_folder=args.image_folder,
        label_file=args.label_file,
        filter_labels=args.filter_labels,
        padding=args.padding,
        output_folder=args.output_folder,
        verbose=args.verbose)
    cutter()

    end = time.time()
    print(f'Total time: {end - start:.2f} s')


class AnotationCutter:
    def __init__(self, image_folder: str, label_file: str,
                 filter_labels: list[str], output_folder: str,
                 padding: int = 0, verbose: bool = False):
        self.image_folder = image_folder
        self.label_file = label_file
        self.filter_labels = [label.lower() for label in filter_labels]
        self.padding = padding
        self.output_folder = output_folder
        self.verbose = verbose

        if verbose:
            logging.basicConfig(level=logging.DEBUG, format='[%(levelname)-s]\t- %(message)s')
        else:
            logging.basicConfig(level=logging.INFO,format='[%(levelname)-s]\t- %(message)s')

        self.logger = logging.getLogger(__name__)
        self.logger.debug(f'Filtering labels: {self.filter_labels}')

        self.annotations = LabelStudioResults(label_file)

        # load image names
        self.image_names = os.listdir(image_folder)
        self.logger.debug(f'Loaded {len(self.image_names)} images from {image_folder}')

        self.annotations.filter_tasks_using_images(self.image_names)
        self.logger.debug(f'Found {len(self.annotations)} images in annotation file')

        if len(self.annotations) == 0:
            raise ValueError(f'No images from annotation file found in image folder {image_folder}')

        os.makedirs(output_folder, exist_ok=True)

    def __call__(self):
        self.logger.info(f'Cutting out objects from {len(self.annotations)} images and saving them to {self.output_folder}')

        # cut out objects from images
        for task in tqdm(self.annotations):
            image_path = task['data']['image']
            img_name = os.path.basename(image_path)
            img = cv2.imread(os.path.join(self.image_folder, img_name))
            img_ext = re.search(r'\.(.+)$', img_name).group(1)
            img_name = img_name.replace(f'.{img_ext}', '')

            results = []
            for annotation in task['annotations']:
                results += annotation['result']

            for result in results:
                if self.filter_labels:
                    labels_filtered = [label for label in result['value']['rectanglelabels'] if label.lower() in self.filter_labels]
                else:
                    labels_filtered = result['value']['rectanglelabels']

                if len(labels_filtered) == 0:
                    self.logger.debug(f'No label for result {result["id"]} in task {task["id"]}')
                    continue

                if len(labels_filtered) > 1:
                    self.logger.warning(f'More than one label for result {result["id"]} in task {task["id"]}.\nUsing only the first one: {labels_filtered[0]}. Other labels: {labels_filtered[1:]}')

                first_label = labels_filtered[0].replace(' ', '_')

                x, y, w, h = label_studio_coords_to_xywh(result['value'], img.shape[:2], padding=self.padding)

                crop = img[y:y+h, x:x+w]
                filename = f"{img_name}_{first_label}_{result['id']}.{img_ext}"
                cv2.imwrite(os.path.join(self.output_folder, filename), crop)


if __name__ == "__main__":
    main()

