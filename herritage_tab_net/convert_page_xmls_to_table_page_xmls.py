#!/usr/bin/python3
"""Load PAGE XML files with table structure in custom tags, convert to table page XML files.

Usage:
$ python3 convert_table....py -x <xml_folder> -o <output_folder>
Resulting in:
    - <output_folder> with PAGE XML layouts with cell information in textline custom tag
    - <output_folder>/render with rendered images with textlines
"""

import argparse
import re
import sys
import os
import time
import logging
import cv2
from tqdm import tqdm
from bs4 import BeautifulSoup
from copy import deepcopy
from collections import defaultdict
import re
import json

import numpy as np

# add parent directory to python file path to enable imports
file_dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_dirname)
sys.path.append(os.path.dirname(file_dirname))

from dataset.label_studio_utils import LabelStudioResults, label_studio_coords_to_xywh, add_padding
from pero_ocr.core.layout import TextLine, PageLayout
from organizer.tables.table_layout import TableCell, TableRegion, TablePageLayout
from organizer.tables.rendering import render_cells, render_table_reconstruction
from organizer.utils import xywh_to_polygon


def parseargs():
    """Parse arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--image-folder", default='example_data/5_table_page_xmls_to_page_xmls/0_images',
        help="Input folder where to look for images.")
    # parser.add_argument(
    #     "-l", "--label-file", type=str, default='example_data/4_annotated_HTML_tables.json',
    #     help="Label studio JSON export file.")
    parser.add_argument(
        "-x", "--xml-folder", type=str, default='example_data/5_table_page_xmls_to_page_xmls/3_page_xmls_with_OCR/',
        help="Folder with table page xml files with detected cells and table structure.")
    parser.add_argument(
        "-o", "--output-folder", type=str, default='example_data/5_table_page_xmls_to_page_xmls/4_back_to_table_page_xmls/',
        help="Output folder where to save cut out objects.")
    # parser.add_argument(
    #     '-v', "--verbose", action='store_true', default=False,
    #     help="Activate verbose logging.")

    return parser.parse_args()


def main():
    args = parseargs()
    print(f'Running {os.path.basename(__file__)} with args: {args}\n{80*"-"}\n')
    start = time.time()

    os.makedirs(args.output_folder, exist_ok=True)
    render_dir = os.path.join(args.output_folder, 'render')
    os.makedirs(render_dir, exist_ok=True)

    xml_files = [xml_file for xml_file in os.listdir(args.xml_folder) if xml_file.endswith('.xml')]
    if len(xml_files) == 0:
        raise ValueError(f'No XML files found in {args.xml_folder}')

    xml_files_exported = 0

    for xml_file in tqdm(xml_files, unit='xml file'):
        xml_file_path = os.path.join(args.xml_folder, xml_file)
        if not os.path.exists(xml_file_path):
            raise ValueError(f'XML file {xml_file} does not exist in {args.xml_folder}')
        if not os.path.isfile(xml_file_path):
            raise ValueError(f'XML file {xml_file} is not a file in {args.xml_folder}')

        page_layout = PageLayout(id=xml_file, file=xml_file_path)
        if page_layout is None:
            raise ValueError(f'Failed to load XML file {xml_file} as PageLayout')
        
        table_layout = TablePageLayout.from_page_layout_with_structure_in_custom(page_layout)

        # render layout to render_file_path
        canvas = load_canvas(xml_file_path, args.image_folder)
        table_layout.render_to_image(canvas)
        render_file_path = os.path.join(render_dir, xml_file.replace('.xml', '.png'))
        cv2.imwrite(render_file_path, canvas)

        # save layout to XML file
        layout_file = os.path.join(args.output_folder, xml_file)
        # page_layout.to_pagexml(layout_file)
        table_layout.to_table_pagexml(layout_file)
        if not os.path.exists(layout_file):
            raise ValueError(f'Failed to save XML file {layout_file}')
        if not os.path.isfile(layout_file):
            raise ValueError(f'XML file {layout_file} is not a file in {args.output_folder}')
        xml_files_exported += 1


    print(f'Exported {xml_files_exported} XML files to {args.output_folder}')
    end = time.time()
    print(f'Total time: {end - start:.2f} s')

def load_canvas(xml_file, image_folder):
    """Load canvas from XML file."""
    fallback_canvas = np.zeros((1000, 1000, 3), dtype=np.uint8)

    # load image
    image_file = os.path.join(image_folder, os.path.basename(xml_file).replace('.xml', '.jpg'))
    if not os.path.exists(image_file) or not os.path.isfile(image_file):
        return fallback_canvas

    image = cv2.imread(image_file)
    return image

if __name__ == "__main__":
    main()

