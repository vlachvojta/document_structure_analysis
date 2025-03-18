#!/usr/bin/python3
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

# add parent directory to python file path to enable imports
file_dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_dirname)
sys.path.append(os.path.dirname(file_dirname))

# from dataset.label_studio_utils import get_label_studio_coords
from pubtables_1m.voc_xml import VocLayout, VocObject, ObjectCategory, VocWord


def parseargs():
    """Parse arguments."""
    print('')
    print('sys.argv: ')
    print(' '.join(sys.argv))
    print('--------------------------------------')

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--image-folder", default='example_data/images_orig',
        help="Input folder where to look for images.")
    parser.add_argument(
        "-x", "--xml-folder", type=str, default='example_data/voc_xml',
        help="Input folder where to look for xml files if PASCOL VOC format.")
    parser.add_argument(
        "-w", "--word-folder", type=str, default='example_data/words',
        help="Input folder where to look for words json files.")
    # parser.add_argument(
    #     "-t", "--task-image-path", type=str, default='/data/local-files/?d=tables/tables_2nd_phase_cell_detection/images/',
    #     help="Input folder where to look for images.")
    # parser.add_argument(
    #     '-p', '--padding', type=int, default=0,
    #     help="Padding around the object.")
    parser.add_argument(
        "-o", "--output-folder", type=str, default='example_data',
        help="Output folder where to save json tasks.")
    parser.add_argument(
        '-v', "--verbose", action='store_true', default=False,
        help="Activate verbose logging.")

    return parser.parse_args()


def main():
    """Main function for simple testing"""
    args = parseargs()

    start = time.time()

    task_creator = TablePartRenderer(
        image_folder=args.image_folder,
        xml_folder=args.xml_folder,
        word_folder=args.word_folder,
        # task_image_path=args.task_image_path,
        output_folder=args.output_folder,
        verbose=args.verbose)
    task_creator()

    end = time.time()
    print(f'Total time: {end - start:.2f} s')


class TablePartRenderer:
    def __init__(self, image_folder: str, xml_folder: str,
                 word_folder: str,
                 output_folder: str, verbose: bool = False):
        self.image_folder = image_folder
        self.xml_folder = xml_folder
        self.word_folder = word_folder
        self.output_folder = output_folder
        # self.task_image_path = task_image_path
        self.output_folder_images_render = os.path.join(output_folder, 'images_render')
        self.output_folder_images_words = os.path.join(output_folder, 'images_words')
        self.output_folder_page_xml = os.path.join(output_folder, 'page_xml')
        self.output_folder_page_xml_render = os.path.join(output_folder, 'page_xml_render')
        self.output_folder_reconstruction = os.path.join(output_folder, 'reconstruction')
        self.output_folder_table_cutouts = os.path.join(output_folder, 'table_cutouts')
        self.verbose = verbose

        if verbose:
            logging.basicConfig(level=logging.DEBUG, format='[%(levelname)-s]\t- %(message)s')
        else:
            logging.basicConfig(level=logging.INFO,format='[%(levelname)-s]\t- %(message)s')

        # TODO check if image and word exists for each xml file
        # TODO allow other images than jpg
        # self.xml_files, self.image_names = self.load_image_xml_pairs(xml_folder, image_folder)
        self.xml_files = [os.path.join(xml_folder, f) for f in os.listdir(xml_folder) if f.endswith('.xml')]
        self.image_names = [f.replace('.xml', '.jpg') for f in os.listdir(xml_folder)]
        self.word_files = [f.replace('.xml', '_words.json') for f in os.listdir(xml_folder)]
        # print(f'words: {self.word_files}')
        logging.debug(f'Loaded {len(self.xml_files)} image-xml pairs')

        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(self.output_folder_images_render, exist_ok=True)
        os.makedirs(self.output_folder_images_words, exist_ok=True)
        os.makedirs(self.output_folder_page_xml, exist_ok=True)
        os.makedirs(self.output_folder_page_xml_render, exist_ok=True)
        os.makedirs(self.output_folder_reconstruction, exist_ok=True)
        os.makedirs(self.output_folder_table_cutouts, exist_ok=True)

        self.categories_seen = set()
        # create stats as a int defaul dict
        self.stats = defaultdict(int)

    def __call__(self):
        print(f'Rendering {len(self.xml_files)} images.')

        for xml_file, image_name, word_file in tqdm(zip(self.xml_files, self.image_names, self.word_files),
                                         total=len(self.xml_files), desc='Rendering images'):
            # print(f'\nParsing {xml_file}')
            image_file = os.path.join(self.image_folder, image_name)
            output_image_file_base = os.path.join(self.output_folder_images_render, image_name)
            # output_words

            voc_layout = VocLayout(xml_file, os.path.join(self.word_folder, word_file))
            if voc_layout is None:
                continue
            layout_categories = set([obj.category for obj in voc_layout.objects])
            self.categories_seen.update(layout_categories)

            image_orig = cv2.imread(image_file)
            if image_orig is None:
                logging.error(f'Could not load image: {image_file}')
                continue

            rendered_words = TablePartRenderer.render_words(image_orig.copy(), voc_layout.words)
            output_file = output_image_file_base.replace('.jpg', '_words.jpg')
            cv2.imwrite(output_file, rendered_words)
            self.stats['total_images_exported'] += 1
            self.stats['words_images_exported'] += 1

            for category in layout_categories:
                image = TablePartRenderer.render_table_parts(image_orig.copy(), voc_layout, [category])

                if image is None:
                    continue

                output_file = output_image_file_base.replace('.jpg', f'_{category.value.replace(" ", "_")}.jpg')
                cv2.imwrite(output_file, image)
                logging.info(f'Saved image with table parts to: {output_file}')
                self.stats['total_images_exported'] += 1
                self.stats[f'{category.value.replace(" ", "_")}_category_images_exported'] += 1

            # render tsr image only with xml defined table objects
            rendered_tsr = voc_layout.render_tsr(image_orig.copy())
            output_file = output_image_file_base.replace('.jpg', '_tsr.jpg')
            cv2.imwrite(output_file, rendered_tsr)
            self.stats['tsr_images_exported'] += 1

            # page layout to image and xml
            table_layout = voc_layout.to_table_layout()
            page_xml_file = os.path.join(self.output_folder_page_xml, image_name.replace('.jpg', '.xml'))
            table_layout.to_table_pagexml(page_xml_file)
            self.stats['page_layouts_exported'] += 1

            rendered_page_layout = table_layout.render_to_image(image_orig.copy(), thickness=1)
            output_file = os.path.join(self.output_folder_page_xml_render, image_name.replace('.jpg', '.jpg'))
            cv2.imwrite(output_file, rendered_page_layout)
            self.stats['page_layout_rendered_exported'] += 1

            rendered_page_layout_cropped = table_layout.render_table_crops(image_orig.copy(), thickness=1)[0]
            output_file = os.path.join(self.output_folder_page_xml_render, image_name.replace('.jpg', '_crop.jpg'))
            cv2.imwrite(output_file, rendered_page_layout_cropped)
            self.stats['page_layout_crops_exported'] += 1

            # render table cutouts
            rendered_page_layout_cropped = table_layout.render_table_crops(image_orig.copy(), thickness=1, render_borders=False)[0]
            output_file = os.path.join(self.output_folder_table_cutouts, image_name)
            cv2.imwrite(output_file, rendered_page_layout_cropped)
            self.stats['table_cutouts_exported'] += 1

            # render table reconstruction
            reconstructed_image = render_table_reconstruction(image_orig.copy(), table_layout.tables[0].cells)
            output_file = os.path.join(self.output_folder_reconstruction, image_name)
            cv2.imwrite(output_file, reconstructed_image)
            self.stats['reconstruction_images_exported'] += 1



        print(f'Categories seen: {self.categories_seen}')
        print(f'Statistics: {json.dumps(self.stats, indent=4)}')

    @staticmethod
    def load_voc_xml(xml_file: str) -> ET.Element:
        """Load VOC XML file."""
        tree = ET.parse(xml_file)
        return tree.getroot()

    @staticmethod
    def render_table_parts(image: np.ndarray, voc_layout: VocLayout, categories: list[ObjectCategory] = None
                           ) -> np.ndarray:
        """Load image and xml file. Render table parts to the image."""
        object_rendered = 0
        for object in voc_layout.objects:
            category = object.category

            if categories and category not in categories:
                continue

            l, t, r, b = object.ltrb()
            cv2.rectangle(image, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 1)
            cv2.putText(image, category.value, (int(l), int(t)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
            object_rendered += 1

        if object_rendered == 0:
            return None
        return image

    @staticmethod
    def render_words(image: np.ndarray, words: list[dict]) -> np.ndarray:
        """Load image and xml file. Render words to the image."""

        for word in words:
            l, t, r, b = word.ltrb()
            cv2.rectangle(image, (int(l), int(t)), (int(r), int(b)), (0, 255, 0), 1)
            # cv2.putText(image, word.text, (int(l), int(t)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        return image


if __name__ == '__main__':
    main()



