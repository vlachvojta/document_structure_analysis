#!/usr/bin/python3
"""Load label studio JSON results and render detected cells with order number guessed by simple algorithm.

Name rendered images same as input images, but with '_order' suffix before the extension.

Usage:
$ python3 order_cell_annotations.py -i <image_folder> -l <label_file> -o <output_folder>
Resulting in <output_folder> with rendered images, named as <image_name>_order.<image_extension>.
"""

import argparse
import re
import sys
import os
import time
import logging
import cv2
from tqdm import tqdm

import numpy as np

# add parent directory to python file path to enable imports
file_dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_dirname)
sys.path.append(os.path.dirname(file_dirname))

from dataset.label_studio_utils import LabelStudioResults, label_studio_coords_to_xywh, add_padding
from organizer.tables.table_layout import TableCell, TableRegion, TablePageLayout
from organizer.tables.order_guessing import guess_order_of_cells, reorder_cells
from organizer.tables.rendering import render_cells
from organizer.utils import xywh_to_polygon


def parseargs():
    """Parse arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--image-folder", default='example_data/2_cell_detection_tasks/images',
        help="Input folder where to look for images.")
    parser.add_argument(
        "-l", "--label-file", type=str, default='example_data/3_annotated_cell_detection.json',
        help="Label studio JSON export file.")
    # parser.add_argument(
    #     '-f', '--filter-labels', nargs='+', default=[],
    #     help="Filter labels to keep. Default: all labels.")
    # parser.add_argument(
    #     '-p', '--padding', type=int, default=0,
    #     help="Padding around the object.")
    parser.add_argument(
        "-o", "--output-folder", type=str, default='example_data/3_cell_detection_order',
        help="Output folder where to save cut out objects.")
    parser.add_argument(
        '-v', "--verbose", action='store_true', default=False,
        help="Activate verbose logging.")

    return parser.parse_args()


def main():
    args = parseargs()
    print(f'Running {os.path.basename(__file__)} with args: {args}\n{80*"-"}\n')

    start = time.time()

    renderer = CellOrderRenderer(
        image_folder=args.image_folder,
        label_file=args.label_file,
        # filter_labels=args.filter_labels,
        # padding=args.padding,
        output_folder=args.output_folder,
        verbose=args.verbose)
    renderer()

    end = time.time()
    print(f'Total time: {end - start:.2f} s')


class CellOrderRenderer:
    def __init__(self, image_folder: str, label_file: str,
                 output_folder: str, # filter_labels: list[str], padding: int = 0, 
                 verbose: bool = False):
        self.image_folder = image_folder
        self.label_file = label_file
        # self.filter_labels = [label.lower() for label in filter_labels]
        # self.padding = padding
        self.output_folder = output_folder
        self.verbose = verbose

        self.output_folder_render = os.path.join(output_folder, 'render')
        self.output_folder_double = os.path.join(output_folder, 'double')
        self.output_folder_xml = os.path.join(output_folder, 'xml')
        # self.output_folder_tasks = os.path.join(output_folder, 'tasks')
        os.makedirs(self.output_folder_render, exist_ok=True)
        os.makedirs(self.output_folder_double, exist_ok=True)
        os.makedirs(self.output_folder_xml, exist_ok=True)
        # os.makedirs(self.output_folder_tasks, exist_ok=True)

        if verbose:
            logging.basicConfig(level=logging.DEBUG, format='[%(levelname)-s]\t- %(message)s')
        else:
            logging.basicConfig(level=logging.INFO,format='[%(levelname)-s]\t- %(message)s')

        self.logger = logging.getLogger(__name__)
        # self.logger.debug(f'Filtering labels: {self.filter_labels}')

        self.annotations = LabelStudioResults(label_file)

        # load image names
        self.image_names = os.listdir(image_folder)
        self.logger.debug(f'Loaded {len(self.image_names)} images from {image_folder}')
        self.annotations.filter_tasks_using_images(self.image_names)
        self.logger.debug(f'Found {len(self.annotations)} images in annotation file')

        if len(self.annotations) == 0:
            raise ValueError(f'No images from annotation file found in image folder {image_folder}')

    def __call__(self):
        self.logger.info(f'Cutting out objects from {len(self.annotations)} images and saving them to {self.output_folder}')

        exported = 0

        for task in tqdm(self.annotations):
            image_path = task['data']['image']
            img_name = os.path.basename(image_path)
            img = cv2.imread(os.path.join(self.image_folder, img_name))
            img_ext = re.search(r'\.(.+)$', img_name).group(1)
            img_name = img_name.replace(f'.{img_ext}', '')
            img_orig = img.copy()

            # get all cells in annotataion task
            cells = self.read_cells_from_task(task, img)

            if len(cells) == 0:
                self.logger.warning(f'No cells found in task {task["id"]}, image {img_name}')
                continue

            # guess order using guess_order_of_cells
            order = guess_order_of_cells(cells)
            # reorder list of cells + put order to IDS so it can be rendered as a text for every cell
            cells = reorder_cells(cells, order)
            # render cells with order number
            img = render_cells(img, cells, render_ids=True)

            filename = f"{img_name}_order.{img_ext}"
            cv2.imwrite(os.path.join(self.output_folder_render, filename), img)

            # create tablePageLayout with one table to export to page-xml
            layout = self.create_layout(cells, img, img_name)
            xml_filename = f"{img_name}.xml"
            xml_path = os.path.join(self.output_folder_xml, xml_filename)
            layout.to_table_pagexml(xml_path)

            # export image with rendered cells and original image
            img = self.create_export_image(img, img_orig)
            filename = f"{img_name}_double.{img_ext}"
            cv2.imwrite(os.path.join(self.output_folder_double, filename), img)

            # test loading the just exported xml file
            layout = TablePageLayout.from_table_pagexml(xml_path)
            assert len(layout.tables) == 1, f'Expected one table in layout, got {len(layout.tables)}'
            loaded_cell_count = layout.tables[0].len(include_faulty=True)
            assert loaded_cell_count == len(cells), f'Expected {len(cells)} cells in table, got {loaded_cell_count}'

            exported += 1

        ratio = exported / len(self.annotations) if len(self.annotations) > 0 else 0
        print(f'Exported {exported} images from {len(self.annotations)} tasks ({ratio:.2%}) '
                         f'to: \n- {self.output_folder_render}\n- {self.output_folder_double}\n- {self.output_folder_xml}')

    def read_cells_from_task(self, task: dict, img: np.ndarray) -> list[TableCell]:
        cells = []

        for annotation in task['annotations']:
            results = annotation['result']

            for result in results:
                x, y, w, h = label_studio_coords_to_xywh(result['value'], img.shape[:2])
                coords = xywh_to_polygon(x, y, w, h)

                labels = result['value']['rectanglelabels']
                if len(labels) == 0:
                    self.logger.debug(f'No label for result {result["id"]} in task {task["id"]}')
                    continue

                if len(labels) > 1:
                    self.logger.warning(f'More than one label for result {result["id"]} in task {task["id"]}.'
                                        f'Using only the first one: {labels[0]}. Other labels: {labels[1:]}')

                cell_category = labels[0].replace(' ', '_')
                table_cell = TableCell(id=result['id'], coords=coords, category=cell_category)
                cells.append(table_cell)

        return cells

    def create_layout(self, cells: list[TableCell], img: np.ndarray, img_name: str) -> TablePageLayout:
        # create tablePageLayout with one table to export to page-xml
        layout = TablePageLayout(id=img_name, file=img_name, page_size=img.shape[:2])
        table_coords = xywh_to_polygon(0, 0, img.shape[1], img.shape[0])
        table = TableRegion(id='table', coords=table_coords)

        for cell in cells:
            cell.faulty = True

        table.faulty_cells.extend(list(cells))  # add all cells as faulty because we don't know the real table structure
        layout.tables = [table]

        return layout
    
    def create_export_image(self, img: np.ndarray, img_orig: np.ndarray) -> np.ndarray:
        # concat orig image with rendered image
        if img.shape[0] < img.shape[1]:
            black_border = np.zeros([10, img_orig.shape[1], 3])
            img = np.vstack([img, black_border, img_orig])
        else:
            black_border = np.zeros([img_orig.shape[0], 10, 3])
            img = np.hstack([img, black_border, img_orig])
        return img

if __name__ == "__main__":
    main()

