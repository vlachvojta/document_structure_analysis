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

# add current working directory + parent to path to enable imports
sys.path.append(os.getcwd())
sys.path.append(os.path.dirname(os.getcwd()))

from dataset.label_studio_results import LabelStudioResults


def parseargs():
    """Parse arguments."""
    print('')
    print('sys.argv: ')
    print(' '.join(sys.argv))
    print('--------------------------------------')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--image-folder", default='example_data/images',
        help="Input folder where to look for images.")
    parser.add_argument(
        "-l", "--label-file", type=str, default='example_data/label_studio_export.json',
        help="Label studio JSON export file.")
    parser.add_argument(
        "-o", "--output-folder", type=str, default='example_data/cut_anotations',
        help="Output folder to copy complete pairs.")
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
        output_folder=args.output_folder,
        verbose=args.verbose)
    cutter()

    end = time.time()
    print(f'Total time: {end - start:.2f} s')


class AnotationCutter:
    def __init__(self, image_folder: str, label_file: str, output_folder: str,
                 verbose: bool = False):
        self.image_folder = image_folder
        self.label_file = label_file
        self.output_folder = output_folder

        if verbose:
            logging.basicConfig(level=logging.DEBUG, format='[%(levelname)-s]\t- %(message)s')
        else:
            logging.basicConfig(level=logging.INFO,format='[%(levelname)-s]\t- %(message)s')

        self.logger = logging.getLogger(__name__)

        # load label studio JSON
        self.annotations = LabelStudioResults(label_file)

        # load image names
        self.image_names = os.listdir(image_folder)
        self.logger.debug(f'Loaded {len(self.image_names)} images from {image_folder}')

        print(self.image_names)
        print(self.annotations.get_images(base_name=True))

        self.annotations.filter_tasks_using_images(self.image_names)
        self.logger.debug(f'Found {len(self.annotations)} images in annotation file')

        if len(self.annotations) == 0:
            raise ValueError(f'No images from annotation file found in image folder {image_folder}')

        # create output folder
        os.makedirs(output_folder, exist_ok=True)

    def __call__(self):
        self.logger.info(f'Cutting out objects from {len(self.annotations)} images and saving them to {self.output_folder}')

        # cut out objects from images
        for task in self.annotations:
            logging.debug(f'Processing task {task["id"]}')
            image_path = task['data']['image']
            img_name = os.path.basename(image_path)
            img = cv2.imread(os.path.join(self.image_folder, img_name))

            img_ext = re.search(r'\.(.+)$', img_name).group(1)
            img_name = img_name.replace(f'.{img_ext}', '')

            for result in task['annotations'][0]['result']:
                x, y, w, h = result['value']['x'], result['value']['y'], result['value']['width'], result['value']['height']

                # Convert percentages to pixels
                x, y, w, h = int(x / 100 * img.shape[1]), int(y / 100 * img.shape[0]), \
                            int(w / 100 * img.shape[1]), int(h / 100 * img.shape[0])
                
                first_label = result['value']['rectanglelabels'][0]
                if len(result['value']['rectanglelabels']) > 1:
                    self.logger.warning(f'More than one label for object {result["id"]}, using only the first one: {first_label}. Other labels: {result["value"]["rectanglelabels"]}')

                crop = img[y:y+h, x:x+w]
                # create filename using this naming convention: <image_name>_<object_label>_<object_id>.<image_extension>
                filename = f"{img_name}_{first_label}_{result['id']}.{img_ext}"
                #filename = f"{img_name}_{result['id']}.jpg"
                cv2.imwrite(os.path.join(self.output_folder, filename), crop)


if __name__ == "__main__":
    main()

