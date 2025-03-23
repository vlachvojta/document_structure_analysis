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
        "-m", "--mass-export", action='store_true', default=False,
        help="Mass export only image crops and PAGE XMLs.")
    parser.add_argument(
        "-f", "--force_new", action='store_true', default=False,
        help="Force new export of all images and xmls.")
    parser.add_argument(
        "-l", "--limit", type=int, default=None,
        help="Limit the number of files to process.")
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

    pubtables_converter = PubTablesConverter(
        image_folder=args.image_folder,
        xml_folder=args.xml_folder,
        word_folder=args.word_folder,
        # task_image_path=args.task_image_path,
        mass_export=args.mass_export,
        force_new=args.force_new,
        limit=args.limit,
        output_folder=args.output_folder,
        verbose=args.verbose)
    pubtables_converter()

    end = time.time()
    print(f'Total time: {end - start:.2f} s')


class PubTablesConverter:
    def __init__(self, image_folder: str, xml_folder: str,
                 word_folder: str, mass_export: bool, force_new: bool,
                 limit: int,
                 output_folder: str, verbose: bool = False):
        self.image_folder = image_folder
        self.xml_folder = xml_folder
        self.word_folder = word_folder
        self.mass_export = mass_export
        self.force_new = force_new
        self.limit = limit
        self.output_folder = output_folder
        # self.task_image_path = task_image_path
        self.output_folder_images_render = os.path.join(output_folder, 'images_render')
        self.output_folder_images_words = os.path.join(output_folder, 'images_words')
        self.output_folder_page_xml = os.path.join(output_folder, 'page_xml')
        self.output_folder_page_xml_render = os.path.join(output_folder, 'page_xml_render')
        self.output_folder_reconstruction = os.path.join(output_folder, 'reconstruction')
        self.output_folder_table_crops = os.path.join(output_folder, 'table_crops')
        self.verbose = verbose

        if verbose:
            logging.basicConfig(level=logging.DEBUG, format='[%(levelname)-s]\t- %(message)s')
        else:
            logging.basicConfig(level=logging.INFO,format='[%(levelname)-s]\t- %(message)s')

        # self.xml_files, self.image_names = self.load_image_xml_pairs(xml_folder, image_folder)
        exts = ['.xml', '.jpg', '_words.json']
        file_groups = load_file_groups([xml_folder, image_folder, word_folder], exts, limit)
        if not file_groups:
            logging.error('No matching files found in the input folders.')
            return

        self.xml_files = [os.path.join(xml_folder, f + exts[0]) for f in file_groups]
        self.image_names = [os.path.join(image_folder, f + exts[1]) for f in file_groups]
        self.word_files = [os.path.join(word_folder, f + exts[2]) for f in file_groups]
        # print(f'words: {self.word_files}')
        logging.debug(f'Loaded {len(self.xml_files)} xml-image-words triplets.')

        os.makedirs(output_folder, exist_ok=True)
        os.makedirs(self.output_folder_page_xml, exist_ok=True)
        os.makedirs(self.output_folder_table_crops, exist_ok=True)
        os.makedirs(self.output_folder_page_xml_render, exist_ok=True)

        if not self.mass_export:
            os.makedirs(self.output_folder_images_render, exist_ok=True)
            os.makedirs(self.output_folder_images_words, exist_ok=True)
            os.makedirs(self.output_folder_reconstruction, exist_ok=True)
            self.categories_seen = set()
        
        self.stats_file = os.path.join(self.output_folder, 'stats.json')

        if os.path.exists(self.stats_file) and not self.force_new and self.mass_export:
            # copy existing stats to a new file
            create_backup(self.stats_file)

        self.warning_stats_file = os.path.join(self.output_folder, 'warning_stats.json')
        if os.path.exists(self.warning_stats_file) and not self.force_new and self.mass_export:
            # copy existing stats to a new file
            create_backup(self.warning_stats_file)

        self.stats = defaultdict(int)
        self.warning_stats = {
            'by_file': {},
            'warning_counts': defaultdict(int),
        }

    def __call__(self):
        print(f'Converting {len(self.xml_files)} triplets of xml, image and words to images.')

        for file_id, (xml_file, image_file, word_file) in enumerate(tqdm(zip(self.xml_files, self.image_names, self.word_files),
                                         total=len(self.xml_files), desc='Rendering images')):
            # print(f'\nParsing {xml_file}')
            if file_id and file_id % 100 == 0:
                self.save_stats()
            image_name = os.path.basename(image_file)
            output_image_file_base = os.path.join(self.output_folder_images_render, image_name)

            output_page_xml_file = os.path.join(self.output_folder_page_xml, image_name.replace('.jpg', '.xml'))
            output_table_crop_file = os.path.join(self.output_folder_table_crops, image_name)
            rendered_page_layout_out_file = os.path.join(self.output_folder_page_xml_render, image_name.replace('.jpg', '.jpg'))
            if (not self.force_new and self.mass_export and
                os.path.exists(output_page_xml_file) and os.path.exists(output_table_crop_file)):
                self.stats['skipped because exists'] += 1
                continue

            voc_layout = VocLayout(xml_file, word_file)
            if voc_layout is None:
                self.stats['voc_layout_load_failed'] += 1
                logging.error(f'Could not load VOC layout: {xml_file}')
                continue

            image_orig = cv2.imread(image_file)
            if image_orig is None:
                self.stats['image_load_failed'] += 1
                logging.error(f'Could not load image: {image_file}')
                continue

            # page layout to image and xml
            table_layout = voc_layout.to_table_layout()
            pagexml_result = table_layout.to_table_pagexml(output_page_xml_file)
            if pagexml_result:
                self.stats['page_layouts_exported'] += 1
            else:
                self.stats['page_layouts_export_failed'] += 1
                self.warning_stats['by_file'][xml_file] = ['page_layout_export_failed']
                self.warning_stats['warning_counts']['page_layout_export_failed'] += 1

            if len(voc_layout.warnings_sent) > 0:
                self.warning_stats['by_file'][xml_file] = voc_layout.warnings_sent
                for warning in voc_layout.warnings_sent:
                    self.warning_stats['warning_counts'][warning] += 1

            if not self.mass_export or len(voc_layout.warnings_sent) > 0:
                # render page layout with table, cell, line and word bounding boxes
                rendered_page_layout = table_layout.render_to_image(image_orig.copy(), thickness=1, circles=False)
                cv2.imwrite(rendered_page_layout_out_file, rendered_page_layout)
                self.stats['page_layout_rendered_exported'] += 1

            # render table cutouts
            table_crop = table_layout.render_table_crops(image_orig.copy(), thickness=1, render_borders=False)[0]
            cv2.imwrite(output_table_crop_file, table_crop)
            self.stats['table_crops_exported'] += 1

            if self.mass_export:
                continue

            layout_categories = set([obj.category for obj in voc_layout.objects])
            self.categories_seen.update(layout_categories)

            # render table reconstruction only if mass export is not enabled
            reconstructed_image = render_table_reconstruction(image_orig.copy(), table_layout.tables[0].cells)
            output_file = os.path.join(self.output_folder_reconstruction, image_name)
            cv2.imwrite(output_file, reconstructed_image)
            self.stats['reconstruction_images_exported'] += 1

            # render words
            rendered_words = PubTablesConverter.render_words(image_orig.copy(), voc_layout.words)
            output_file = output_image_file_base.replace('.jpg', '_words.jpg')
            cv2.imwrite(output_file, rendered_words)
            self.stats['total_images_exported'] += 1
            self.stats['words_images_exported'] += 1

            # render table parts
            for category in layout_categories:
                image = PubTablesConverter.render_table_parts(image_orig.copy(), voc_layout, [category])

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

            rendered_page_layout_cropped = table_layout.render_table_crops(rendered_page_layout, render_borders=False)[0]
            output_file = os.path.join(self.output_folder_page_xml_render, image_name.replace('.jpg', '_crop.jpg'))
            cv2.imwrite(output_file, rendered_page_layout_cropped)
            self.stats['page_layout_crops_exported'] += 1

        self.save_stats()
        print('')
        # print(f'Categories seen: {self.categories_seen}')
        print(f'Statistics: {json.dumps(self.stats, indent=4)}')
        print(f'Warnings: {json.dumps(self.warning_stats["warning_counts"], indent=4)}')
        print(f'(more stats in {self.output_folder}: {os.path.basename(self.stats_file)}, {os.path.basename(self.warning_stats_file)})')

    def save_stats(self):
        with open(self.stats_file, 'w') as f:
            json.dump(self.stats, f, indent=4)
        with open(self.warning_stats_file, 'w') as f:
            json.dump(self.warning_stats, f, indent=4)

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

def load_file_groups(folders: list[str]=['example_data/voc_xml', 'example_data/images_orig', 'example_data/words'], exts: list[str]=['.xml', '.jpg', '_words.json'], limit: int = None) -> list[str]:
    if not folders:
        return []

    print(f'Loading files from folders:')

    folder_files: list(set) = []
    for folder, ext in zip(folders, exts):
        files = [f.replace(ext, '') for f in os.listdir(folder) if f.endswith(ext)]
        folder_files.append(set(files))
        print(f'\t{folder}: {len(files)} files')

    interestction = sorted(list(set.intersection(*folder_files)))
    print(f'\tInterestction: \t{len(interestction)} files')

    if limit:
        interestction = interestction[:limit]

    return interestction

def create_backup(file: str):
    # split file name and extension (extension is whatever there is after the last dot)
    filename, file_extension = os.path.splitext(file)
    backup_file = filename + '_backup_' + time.strftime("%Y%m%d-%H%M%S") + file_extension
    os.rename(file, backup_file)

if __name__ == '__main__':
    main()



