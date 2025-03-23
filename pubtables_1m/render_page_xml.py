#!/usr/bin/python3
# TODO update docs for this file
"""Load VOC XML + words JSON from pubtables-1m dataset. Render table parts to images, export page XML and reconstruction.

Usage:
$ python3 ocr_to_tasks.py -i <image_folder> -x <xml_folder> -t <task_image_path> -o <output_folder>
Resulting in <output_folder>/* with output stuff. See example_data for example input and output data.
"""

import argparse
import sys
import os
import time
import logging
import json
from tqdm import tqdm
import lxml.etree as ET
from collections import defaultdict

import cv2
import numpy as np

# from pero_ocr.core.layout import PageLayout
from organizer.tables.rendering import render_table_reconstruction
from organizer.tables.table_layout import TablePageLayout, TableRegion, TableCell

# add parent directory to python file path to enable imports
file_dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_dirname)
sys.path.append(os.path.dirname(file_dirname))

# from dataset.label_studio_utils import get_label_studio_coords
from pubtables_1m.pubtables_converter import load_file_groups, create_backup


def parseargs():
    """Parse arguments."""
    print('')
    print('sys.argv: ')
    print(' '.join(sys.argv))
    print('--------------------------------------')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--image-folder", default='example_data/table_crops',
        help="Input folder where to look for images.")
    parser.add_argument(
        "-x", "--xml-folder", type=str, default='example_data/page_xml',
        help="PAGE XML folder where to look for xml files.")
    parser.add_argument(
        "-w", "--word-folder", type=str, default='example_data/words',
        help="Input folder where to look for words json files.")
    parser.add_argument(
        "-l", "--limit", type=int, default=None,
        help="Limit the number of files to process.")
    parser.add_argument(
        "-o", "--output-folder", type=str, default='example_data/page_xml_rendered_with_render_page_xml.py',
        help="Output folder where to save json tasks.")
    parser.add_argument(
        '-v', "--verbose", action='store_true', default=False,
        help="Activate verbose logging.")

    return parser.parse_args()


def main():
    """Main function for simple testing"""
    args = parseargs()

    start = time.time()

    page_xml_renderer = PageXmlRenderer(
        image_folder=args.image_folder,
        xml_folder=args.xml_folder,
        limit=args.limit,
        output_folder=args.output_folder,
        verbose=args.verbose)
    page_xml_renderer()

    end = time.time()
    print(f'Total time: {end - start:.2f} s')


class PageXmlRenderer:
    def __init__(self, image_folder: str, xml_folder: str,
                #  word_folder: str, mass_export: bool, force_new: bool,
                 limit: int,
                 output_folder: str, verbose: bool = False):
        self.image_folder = image_folder
        self.xml_folder = xml_folder
        self.output_folder = output_folder
        # self.task_image_path = task_image_path
        self.verbose = verbose

        if verbose:
            logging.basicConfig(level=logging.DEBUG, format='[%(levelname)-s]\t- %(message)s')
        else:
            logging.basicConfig(level=logging.INFO,format='[%(levelname)-s]\t- %(message)s')

        # self.xml_files, self.image_names = self.load_image_xml_pairs(xml_folder, image_folder)
        exts = ['.xml', '.jpg']
        file_groups = load_file_groups([xml_folder, image_folder], exts, limit)
        if not file_groups:
            logging.error('No matching files found in the input folders.')
            return

        self.xml_files = [os.path.join(xml_folder, f + exts[0]) for f in file_groups]
        self.image_names = [os.path.join(image_folder, f + exts[1]) for f in file_groups]
        # print(f'words: {self.word_files}')
        logging.debug(f'Loaded {len(self.xml_files)} xml-image pairs.')

        os.makedirs(output_folder, exist_ok=True)

    def __call__(self):
        print(f'Rendering {len(self.xml_files)} images.')

        for file_id, (xml_file, image_file) in enumerate(tqdm(zip(self.xml_files, self.image_names),
                                                         total=len(self.xml_files), desc='Rendering images')):
            # print(f'\nParsing {xml_file}')
            image_name = os.path.basename(image_file)
            output_image_file_base = os.path.join(self.output_folder, image_name)

            # load page xml from xml_file
            page_layout = TablePageLayout.from_table_pagexml(xml_file)
            if page_layout is None:
                print(f'Could not load page layout from xml file: {xml_file}')
                continue

            # load input image
            image_orig = cv2.imread(image_file)
            if image_orig is None:
                print(f'Could not load image: {image_file}')
                continue

            # render page layout
            image_render = page_layout.render_to_image(image_orig.copy(), thickness=1, circles=False)
            cv2.imwrite(output_image_file_base, image_render)

if __name__ == '__main__':
    main()



