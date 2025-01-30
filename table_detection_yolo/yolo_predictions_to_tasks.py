#!/usr/bin/python3
"""This script creates Label Studio tasks from YOLO predictions.

Usage:
$ python3 yolo_predictions_to_tasks.py -i <image_folder> -x <prediction_folder> -t <task_image_path> -o <output_folder>
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
import csv

import numpy as np
import cv2

# add parent directory to python file path to enable imports
file_dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_dirname)
sys.path.append(os.path.dirname(file_dirname))

from dataset import label_studio_utils as utils
from pero_ocr.core.layout import PageLayout


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
        "-p", "--prediction-folder", type=str, default='example_data/predictions',
        help="Input folder where to look for images.")
    parser.add_argument(
        "-t", "--task-image-path", type=str, default='/data/local-files/?d=tables/tables_1st_phase_detection/images_with_predictions/04',
        help="Input folder where to look for images.")
    parser.add_argument(
        "-o", "--output-folder", type=str, default='example_data/tasks',
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
        prediction_folder=args.prediction_folder,
        task_image_path=args.task_image_path,
        output_folder=args.output_folder,
        verbose=args.verbose)
    task_creator()

    end = time.time()
    print(f'Total time: {end - start:.2f} s')


class TaskCreator:
    def __init__(self, image_folder: str, prediction_folder: str,
                 task_image_path: str, output_folder: str,
                 verbose: bool = False):
        self.image_folder = image_folder
        self.prediction_folder = prediction_folder
        self.task_image_path = task_image_path
        self.output_folder = output_folder
        self.output_folder_tasks = os.path.join(output_folder, 'tasks')
        self.output_folder_images = os.path.join(output_folder, 'images')
        self.verbose = verbose

        if verbose:
            logging.basicConfig(level=logging.DEBUG, format='[%(levelname)-s]\t- %(message)s')
        else:
            logging.basicConfig(level=logging.INFO,format='[%(levelname)-s]\t- %(message)s')

        self.prediction_files, self.image_names = self.load_image_prediction_pairs(prediction_folder, image_folder)
        logging.debug(f'Loaded {len(self.prediction_files)} image-prediction pairs')

        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(self.output_folder_tasks, exist_ok=True)
        os.makedirs(self.output_folder_images, exist_ok=True)

    def __call__(self):
        print(f'Creating tasks from {len(self.prediction_files)} image-prediction pairs.')

        # for each image-prediction pair
        # iterate over prediction files and images using tqdn
        total = len(self.prediction_files)
        for prediction_file, image_name in tqdm(zip(self.prediction_files, self.image_names),
                                         total=total, desc='Creating tasks'):
            # load page prediction using PageLayout
            prediction_file_path = os.path.join(self.prediction_folder, prediction_file)
            image_path = os.path.join(self.image_folder, image_name)

            task_name = os.path.splitext(prediction_file)[0] + '.json'
            task_path = os.path.join(self.output_folder_tasks, task_name)

            # read prediction from file
            predictions = self.get_predictions_from_file(prediction_file_path)
            print(f'Loaded predictions from file {prediction_file_path}: {predictions}')

            # read original image size from image_path
            image_loaded = cv2.imread(image_path)
            orig_h, orig_w, _ = image_loaded.shape
            print(f'from image {image_path} loaded image with shape {image_loaded.shape} and orig_h {orig_h} orig_w {orig_w}')

            task = self.create_task(predictions, image_name, orig_h=orig_h, orig_w=orig_w)

            # save json task
            with open(task_path, 'w') as f:
                json.dump(task, f, indent=4)

            # copy image to self.output_folder_images using shutil
            shutil.copy(image_path, self.output_folder_images)

        ls_tasks = os.listdir(self.output_folder_tasks)
        ls_images = os.listdir(self.output_folder_images)
        print(f'Created {len(ls_tasks)} tasks and {len(ls_images)} images in {self.output_folder_tasks} and {self.output_folder_images}')

    def load_image_prediction_pairs(self, prediction_folder: str, image_folder: str) -> tuple[list[str], list[str]]:
        # load prediction files
        prediction_files = os.listdir(prediction_folder)
        print(f'Loaded {len(prediction_files)} prediction files from {prediction_folder}')

        # load image names
        image_names = os.listdir(image_folder)
        print(f'Loaded {len(image_names)} images from {image_folder}')

        # create image prediction pairs, but keep in mind images can have different extensions
        files = {}

        for image_file in image_names:
            striped = os.path.splitext(image_file)[0]

            if striped in files:
                files[striped]['images'].append(image_file)
            else:
                files[striped] = {'prediction': None, 'images': [image_file]}

        for prediction_file in prediction_files:
            striped = os.path.splitext(prediction_file)[0]

            if striped in files:
                files[striped]['prediction'] = prediction_file
            else:
                files[striped] = {'prediction': prediction_file, 'images': []}

        # find common image names
        good_files = {}
        for file in files:
            if len(files[file]['images']) > 1:
                print(f'More than one image for file {file}. '
                                    f'Using only the first one: {files[file]["images"][0]}. '
                                    f'Other images: {files[file]["images"][1:]}')

            if files[file]['prediction'] is None or files[file]['images'] == []:
                # del files[file]
                continue

            good_files[file] = files[file]

        files = good_files

        print(f'Found {len(files)} common names between prediction and image files')

        prediction_files = [files[file]['prediction'] for file in files]
        image_names = [files[file]['images'][0] for file in files]
        return prediction_files, image_names

    def create_task(self, predictions: list, image_name: str, orig_h: int, orig_w: int) -> dict:
        predictions_dicts = []
        for i, prediction in enumerate(predictions):

            # get_label_studio_coords
            print(f'prediction: {prediction["coords"]}, orig_h: {orig_h}, orig_w: {orig_w}')
            x, y, w, h = utils.get_label_studio_coords_from_xywh(prediction['coords'], (orig_h, orig_w))
            print(f'x: {x}, y: {y}, w: {w}, h: {h}')
            print('')

            predictions_dicts.append(
                {
                    'id': f'{i:04d}',
                    'type': 'rectanglelabels',
                    "meta": {
                        "text": prediction['label'],
                    },
                    'from_name': 'label',
                    'to_name': 'image',
                    'original_height': orig_h,
                    'original_width': orig_w,
                    'image_rotation': 0,
                    'value': {
                        'rotation': 0,
                        'x': x,
                        'y': y,
                        'width': w,
                        'height': h,
                        'rectanglelabels': ['Unsure']
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
                    'result': predictions_dicts
                }
            ]
        }

        return task

    def get_predictions_from_file(self, prediction_file: str) -> list[np.ndarray]:
        predictions = []
        with open(prediction_file, 'r') as f:
            reader = csv.reader(f, delimiter=' ')
            for row in reader:
                if not len(row) == 5:
                    print(f'Skipping row with wrong number of columns ({len(row)}): {row}')
                    continue
                label, x, y, x2, y2 = row
                w = float(x2) - float(x)
                h = float(y2) - float(y)

                try:
                    predictions.append({
                        'label': label,
                        'coords': np.array([float(x), float(y), float(w), float(h)], dtype=np.float32)
                    })
                except ValueError as e:
                    print(f'Error parsing row {row}: {e}')
                    continue
        return predictions


if __name__ == "__main__":
    main()

