#!/usr/bin/python3
"""Export Pero-OCR page XML results to Label Studio JSON format with detected textlines as predictions.

Usage:
$ python3 ocr_to_tasks.py -i <image_folder> -x <xml_folder> -t <task_image_path> -o <output_folder>
Resulting in <output_folder>/tasks with tasks and <output_folder>/images with images.
"""

import argparse
import sys
import os
import time
import logging
import json
from tqdm import tqdm
from time import sleep
import shutil

from pero_ocr.core.layout import PageLayout
import numpy as np

# add parent directory to python file path to enable imports
file_dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_dirname)
sys.path.append(os.path.dirname(file_dirname))

from dataset.label_studio_utils import get_label_studio_coords


def parseargs():
    """Parse arguments."""
    print('')
    print('sys.argv: ')
    print(' '.join(sys.argv))
    print('--------------------------------------')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--image-folder", default='example_data/1_cut_tables',
        help="Input folder where to look for images.")
    parser.add_argument(
        "-x", "--xml-folder", type=str, default='example_data/1_cut_tables/xml',
        help="Input folder where to look for images.")
    parser.add_argument(
        "-t", "--task-image-path", type=str, default='/data/local-files/?d=tables/tables_2nd_phase_cell_detection/images/',
        help="Input folder where to look for images.")

    # parser.add_argument(
    #     '-p', '--padding', type=int, default=0,
    #     help="Padding around the object.")
    parser.add_argument(
        "-o", "--output-folder", type=str, default='example_data/2_cell_detection_tasks',
        help="Output folder where to save json tasks.")
    parser.add_argument(
        '-v', "--verbose", action='store_true', default=False,
        help="Activate verbose logging.")

    return parser.parse_args()


def main():
    """Main function for simple testing"""
    args = parseargs()

    start = time.time()

    task_creator = TaskCreator(
        image_folder=args.image_folder,
        xml_folder=args.xml_folder,
        task_image_path=args.task_image_path,
        output_folder=args.output_folder,
        verbose=args.verbose)
    task_creator()

    end = time.time()
    print(f'Total time: {end - start:.2f} s')


class TaskCreator:
    def __init__(self, image_folder: str, xml_folder: str,
                 task_image_path: str, output_folder: str,
                 verbose: bool = False):
        self.image_folder = image_folder
        self.xml_folder = xml_folder
        self.task_image_path = task_image_path
        self.output_folder = output_folder
        self.output_folder_tasks = os.path.join(output_folder, 'tasks')
        self.output_folder_images = os.path.join(output_folder, 'images')
        self.verbose = verbose

        if verbose:
            logging.basicConfig(level=logging.DEBUG, format='[%(levelname)-s]\t- %(message)s')
        else:
            logging.basicConfig(level=logging.INFO,format='[%(levelname)-s]\t- %(message)s')

        self.xml_files, self.image_names = self.load_image_xml_pairs(xml_folder, image_folder)
        logging.debug(f'Loaded {len(self.xml_files)} image-xml pairs')

        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(self.output_folder_tasks, exist_ok=True)
        os.makedirs(self.output_folder_images, exist_ok=True)

    def __call__(self):
        print(f'Creating tasks from {len(self.xml_files)} image-xml pairs.')

        # for each image-xml pair
        # iterate over xml files and images using tqdn
        total = len(self.xml_files)
        for xml_file, image_name in tqdm(zip(self.xml_files, self.image_names),
                                         total=total, desc='Creating tasks'):
            # load page xml using PageLayout
            xml_file_path = os.path.join(self.xml_folder, xml_file)
            image_path = os.path.join(self.image_folder, image_name)

            task_name = os.path.splitext(xml_file)[0] + '.json'
            task_path = os.path.join(self.output_folder_tasks, task_name)

            # create task
            layout = PageLayout(file=xml_file_path)
            task = self.create_task(layout, image_name)

            # save json task
            with open(task_path, 'w') as f:
                json.dump(task, f, indent=4)

            # copy image to self.output_folder_images using shutil
            shutil.copy(image_path, self.output_folder_images)

        ls_tasks = os.listdir(self.output_folder_tasks)
        ls_images = os.listdir(self.output_folder_images)
        print(f'Created {len(ls_tasks)} tasks and {len(ls_images)} images in {self.output_folder_tasks} and {self.output_folder_images}')

    def load_image_xml_pairs(self, xml_folder: str, image_folder: str) -> tuple[list[str], list[str]]:
        # load xml files
        xml_files = os.listdir(xml_folder)
        print(f'Loaded {len(xml_files)} xml files from {xml_folder}')

        # load image names
        image_names = os.listdir(image_folder)
        print(f'Loaded {len(image_names)} images from {image_folder}')

        # create image xml pairs, but keep in mind images can have different extensions
        files = {}

        for image_file in image_names:
            striped = os.path.splitext(image_file)[0]

            if striped in files:
                files[striped]['images'].append(image_file)
            else:
                files[striped] = {'xml': None, 'images': [image_file]}

        for xml_file in xml_files:
            striped = os.path.splitext(xml_file)[0]

            if striped in files:
                files[striped]['xml'] = xml_file
            else:
                files[striped] = {'xml': xml_file, 'images': []}

        # find common image names
        good_files = {}
        for file in files:
            if len(files[file]['images']) > 1:
                print(f'More than one image for file {file}. '
                                    f'Using only the first one: {files[file]["images"][0]}. '
                                    f'Other images: {files[file]["images"][1:]}')

            if files[file]['xml'] is None or files[file]['images'] == []:
                # del files[file]
                continue

            good_files[file] = files[file]

        files = good_files

        print(f'Found {len(files)} common names between xml and image files')

        xml_files = [files[file]['xml'] for file in files]
        image_names = [files[file]['images'][0] for file in files]
        return xml_files, image_names

    def create_task(self, layout: PageLayout, image_name: str) -> dict:
        predictions = []
        for textline in layout.lines_iterator():

            # get_label_studio_coords
            x, y, w, h = get_label_studio_coords(textline.polygon, layout.page_size)

            predictions.append(
                {
                    'id': textline.id,
                    'type': 'rectanglelabels',
                    "meta": {
                        "text": textline.transcription,
                    },
                    'from_name': 'label',
                    'to_name': 'image',
                    'original_height': layout.page_size[0],
                    'original_width': layout.page_size[1],
                    'image_rotation': 0,
                    'value': {
                        'rotation': 0,
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'rectanglelabels': ['Table cell']
                    }
                }
            )

        task = {
            'data': {
                'image': os.path.join(self.task_image_path, image_name)
            },
            'predictions': [
                {
                    'model_version': 'pero-ocr',
                    'score': "1.0",
                    'result': predictions
                }
            ]
        }

        return task


if __name__ == "__main__":
    main()

