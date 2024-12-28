#!/usr/bin/python3
"""Load label studio JSON results + XML pages and render reconstructed tables.
- JSON result: HTML representation of table using cell IDs as a refference.
- XML page: Detected cells with order number guessed by simple algorithm.

Name rendered images same as input images, but with '_reconstructed' suffix before the extension.

Usage:
$ python3 order_cell_annotations.py -i <image_folder>  -x <xml_folder> -l <label_file> -o <output_folder>
Resulting in:
    - <output_folder>/images with rendered images, named as <image_name>_reconstructed.<image_extension>
    - <output_folder>/xml with reconstructed tables in page-xml format
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

import numpy as np

# add parent directory to python file path to enable imports
file_dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_dirname)
sys.path.append(os.path.dirname(file_dirname))

from dataset.label_studio_utils import LabelStudioResults, label_studio_coords_to_xywh, add_padding
from pero_ocr.core.layout import TextLine
from organizer.tables.table_layout import TableCell, TableRegion, TablePageLayout
from organizer.tables.order_guessing import guess_order_of_cells, reorder_cells
from organizer.tables.rendering import render_cells, render_table_reconstruction
from organizer.utils import xywh_to_polygon


def parseargs():
    """Parse arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--image-folder", default='example_data/2_cell_detection_tasks/images',
        help="Input folder where to look for images.")
    parser.add_argument(
        "-x", "--xml-folder", type=str, default='example_data/3_cell_detection_order/xml',
        help="Folder with table page xml files with detected cells (ignore structure even if there is).")
    parser.add_argument(
        "-l", "--label-file", type=str, default='example_data/4_annotated_HTML_tables.json',
        help="Label studio JSON export file.")
    parser.add_argument(
        "-o", "--output-folder", type=str, default='example_data/4_reconstructed_tables',
        help="Output folder where to save cut out objects.")
    parser.add_argument(
        '-v', "--verbose", action='store_true', default=False,
        help="Activate verbose logging.")

    return parser.parse_args()


def main():
    args = parseargs()
    print(f'Running {os.path.basename(__file__)} with args: {args}\n{80*"-"}\n')

    start = time.time()

    constructor = TableConstructor(
        image_folder=args.image_folder,
        xml_folder=args.xml_folder,
        label_file=args.label_file,
        output_folder=args.output_folder,
        verbose=args.verbose)
    constructor()

    end = time.time()
    print(f'Total time: {end - start:.2f} s')


class TableConstructor:
    def __init__(self, image_folder: str, xml_folder: str, label_file: str,
                 output_folder: str, verbose: bool = False):
        self.image_folder = image_folder
        self.xml_folder = xml_folder
        self.label_file = label_file
        self.output_folder = output_folder
        self.verbose = verbose

        self.output_folder_render = os.path.join(output_folder, 'render')
        self.output_folder_reconstrution = os.path.join(output_folder, 'reconstruction')
        self.output_folder_xml = os.path.join(output_folder, 'xml')
        self.output_folder_html = os.path.join(output_folder, 'html')
        os.makedirs(self.output_folder_render, exist_ok=True)
        os.makedirs(self.output_folder_reconstrution, exist_ok=True)
        os.makedirs(self.output_folder_xml, exist_ok=True)
        os.makedirs(self.output_folder_html, exist_ok=True)

        if verbose:
            logging.basicConfig(level=logging.DEBUG, format='[%(levelname)-s]\t- %(message)s')
        else:
            logging.basicConfig(level=logging.INFO,format='[%(levelname)-s]\t- %(message)s')

        self.logger = logging.getLogger(__name__)

        self.annotations = LabelStudioResults(label_file)
        self.annotations.replace_image_names('_double.jpg', '.jpg')

        # load image names + filter tasks using images
        self.image_names = os.listdir(image_folder)
        print(f'Loaded {len(self.image_names)} images from {image_folder}')

        self.xml_names = os.listdir(xml_folder)
        self.simulated_images_from_xml = [os.path.basename(xml).replace('.xml', '.jpg') for xml in self.xml_names]
        print(f'Loaded {len(self.xml_names)} xml files from {xml_folder}')

        filter_images = set(self.image_names) | set(self.simulated_images_from_xml)
        self.annotations.filter_tasks_using_images(filter_images)
        print(f'Found {len(self.annotations)} tasks in annotation file that have corresponding images and xmls.')


        if len(self.annotations) == 0:
            raise ValueError(f'No images from annotation file found in image folder {image_folder}')

    def __call__(self):
        print(f'Reconstructing tables from {len(self.annotations)} images and saving them to {self.output_folder}')

        exported = 0

        for task in tqdm(self.annotations):
            image_path = task['data']['image']
            img_name = os.path.basename(image_path)
            img = cv2.imread(os.path.join(self.image_folder, img_name))
            img_ext = re.search(r'\.(.+)$', img_name).group(1)
            img_name = img_name.replace(f'.{img_ext}', '')
            img_orig = img.copy()

            xml_name = img_name + '.xml'
            layout = TablePageLayout.from_table_pagexml(os.path.join(self.xml_folder, xml_name))
            assert len(layout.tables) == 1, f'Expected one table in layout, got {len(layout.tables)}'

            html = self.read_html_from_task(task)
            if html is None:
                continue

            html_filename = f"{img_name}.html"
            html_path = os.path.join(self.output_folder_html, html_filename)
            with open(html_path, 'w') as f:
                f.write(html.prettify())

            # soup find table
            html_table = html.find('table')
            if html_table is None:
                self.logger.warning(f'No table in task {task["id"]}')
                return None

            table, cells = self.html_table_to_numpy(html_table)
            logging.debug(f'loaded {len(cells)} cells in table {table.shape}.')

            assert len(cells) <= layout.tables[0].len(include_faulty=True), \
                f'Loaded more cells from HTML table than in XML page: {len(cells)} vs {layout.tables[0].len(include_faulty=True)}'

            self.join_layout_cells_to_table_cells(layout, cells)

            layout.tables[0].cells = None
            layout.tables[0].faulty_cells = []
            layout.tables[0].insert_cells(cells)

            # export xml layout with joined cells
            xml_filename = f"{img_name}.xml"
            xml_path = os.path.join(self.output_folder_xml, xml_filename)
            layout.to_table_pagexml(xml_path)

            # render page with joined cells
            img = render_cells(img, layout.tables[0].cell_iterator(include_faulty=True))
            filename = f"{img_name}_render.{img_ext}"
            cv2.imwrite(os.path.join(self.output_folder_render, filename), img)

            # render table reconstruction
            img = img_orig.copy()
            img = render_table_reconstruction(img, layout.tables[0].cells)
            filename = f"{img_name}_reconstruction.{img_ext}"
            cv2.imwrite(os.path.join(self.output_folder_reconstrution, filename), img)

            exported += 1

        ratio = exported / len(self.annotations) if len(self.annotations) > 0 else 0
        print(f'Exported {exported} images from {len(self.annotations)} tasks ({ratio:.2%}) '
                         f'to: \n- {self.output_folder_render}\n- {self.output_folder_reconstrution}\n- {self.output_folder_xml}')

    def read_html_from_task(self, task: dict) -> str:
        if len(task['annotations']) == 0:
            self.logger.warning(f'No annotations in task {task["id"]}')
            return None
        elif len(task['annotations']) > 1:
            lenn = len(task['annotations'])
            self.logger.warning(f'More than one ({lenn}) annotation in task {task["id"]}. Using only the first.')

        task['annotations'] = sorted(task['annotations'], key=lambda x: x['id'])
        annotation = task['annotations'][0]
        if len(annotation['result']) == 0:
            self.logger.warning(f'No results in task {task["id"]}')
            return None
        elif len(annotation['result']) > 1:
            lenn = len(annotation['result'])
            self.logger.warning(f'More than one ({lenn}) result in task {task["id"]}. Using only the first.')
        
        if 'text' not in annotation['result'][0]['value']:
            self.logger.warning(f'No text in task {task["id"]}')
            return None

        html = annotation['result'][0]['value']['text'][0]
        if html is None:
            self.logger.warning(f'No HTML in task {task["id"]}')
            return None

        # beautify html
        soup = BeautifulSoup(html)

        return soup

    def html_table_to_numpy(self, table: BeautifulSoup) -> tuple[np.ndarray, list[TableCell]]:
        max_rows, max_cols = self.get_max_rows_cols(table)
        rows = table.find_all('tr')
        safety_padding = 5  # prevent index out of bounds, is deleted at the end of this function
        cell_repeater_id = -42

        # create numpy array for numbers indicating cell rank in cells list
        table_np = np.zeros((max_rows + safety_padding, max_cols + safety_padding), dtype=int)
        cells = []

        # create cells and fill numpy array with cell ranks
        for i, row in enumerate(rows):
            cols = row.find_all(['td', 'th'])
            j = 0
            for col in cols:
                if col is None:
                    print(f'col is None in cell {i}, {j}, which is strange...... TODO investigate')
                    continue

                if table_np[i, j] == cell_repeater_id:
                    j += 1
                    continue
                elif table_np[i, j] > 0:
                    print(f'cell {i}, {j} already filled')
                    j += 1
                    continue

                cell_ids = self.cell_text_to_ids(col.text)
                if cell_ids is None or len(cell_ids) == 0:
                    j += 1
                    continue

                col_span = int(col.get('colspan', 1))
                row_span = int(col.get('rowspan', 1))
                if col_span > 1 or row_span > 1:
                    self.logger.debug(f'found span: {col_span}x{row_span} in cell {i}, {j}')

                table_np[i:i+row_span, j:j+col_span] = cell_repeater_id  # fill span with repeater id

                if len(cell_ids) == 1:
                    cell_id = cell_ids[0]
                    cell = TableCell(id=cell_id, coords=None, row=i, col=j, row_span=row_span, col_span=col_span)
                    cells.append(cell)
                else:
                    lenn = len(cell_ids)
                    self.logger.debug(f'found more than one ({lenn}) cell id in cell {i}, {j}: {cell_ids}')
                    joined_cell_id = ','.join([str(cell_id) for cell_id in cell_ids])
                    cell = TableCell(id=joined_cell_id, coords=None, row=i, col=j, row_span=row_span, col_span=col_span)
                    cell.lines = [TextLine(id=cell_id, polygon=None) for cell_id in cell_ids]
                    cells.append(cell)

                # save rank of cell in cell list to numpy array
                table_np[i, j] = len(cells) - 1
                j += col_span

        # delete empty rows and columns with only zeros
        table_np = table_np[~np.all(table_np == 0, axis=1)]

        # delete empty columns with only zeros
        table_np = table_np[:, ~np.all(table_np == 0, axis=0)]

        return table_np, cells

    def get_max_rows_cols(self, table: BeautifulSoup) -> tuple[int, int]:
        rows = table.find_all('tr')

        max_cols = 0
        for row in rows:
            cols = row.find_all(['td', 'th'])
            cols_count = 0
            for col in cols:
                col_span = int(col.get('colspan', 1))
                cols_count += col_span

            max_cols = max(max_cols, cols_count)

        return len(rows), max_cols

    def cell_text_to_ids(self, cell_text: str) -> list[str]:
        # split text by new line and remove empty strings
        cell_text = cell_text.replace('\n\n', ',')
        cell_text = cell_text.replace('\n', ',')
        cell_text = cell_text.strip()

        if len(cell_text) == 0:
            return None

        cell_ids = cell_text.split(',')
        cell_ids = [text.strip() for text in cell_ids if text.strip()]

        # try:
        #     cell_ids = [text for text in cell_ids]
        # except ValueError:
        #     raise ValueError(f'Failed to convert cell text to int: {cell_text} with ids: {cell_ids}')

        if len(cell_ids) == 0:
            return None

        return cell_ids

    def cell_polygon_from_lines(self, lines: list[np.ndarray]) -> np.ndarray:
        # get min and max x, y from all lines
        x1 = min([line[:, 0].min() for line in lines])
        y1 = min([line[:, 1].min() for line in lines])
        x2 = max([line[:, 0].max() for line in lines])
        y2 = max([line[:, 1].max() for line in lines])

        w = x2 - x1
        h = y2 - y1

        return xywh_to_polygon(x1, y1, w, h)

    def get_the_most_common_category(self, lines: list[TextLine]) -> str:
        line_categories = [line.category for line in lines]
        return max(set(line_categories), key=line_categories.count)

    def join_layout_cells_to_table_cells(self, layout: TablePageLayout, cells: list[TableCell]):
        """Go through all cells and add coords and category from layout cells."""
        layout_id_to_cell = {cell.id: cell for cell in layout.tables[0].cell_iterator(include_faulty=True)}

        for cell in cells:
            if len(cell.lines) > 0:
                lenn = len(cell.lines)
                self.logger.debug(f'cell with ID {cell.id} has more than one ({lenn}) line: {cell.lines}')
                for line in cell.lines:
                    self.logger.debug(f'adding line: {line.id} in a cell with ID {cell.id}')

                    layout_cell = layout_id_to_cell.get(str(line.id))
                    if layout_cell is None:
                        raise ValueError(f'Cell with ID {line.id} not found in layout cells')

                    # add layout cell coords + category to line
                    line.polygon = layout_cell.coords
                    line.category = layout_cell.category
                # add joined lines coords + category to cell coords
                cell.coords = self.cell_polygon_from_lines([line.polygon for line in cell.lines])
                cell.category = self.get_the_most_common_category(cell.lines)
            else:
                # add layout cell coords and category to cell coords
                layout_cell = layout_id_to_cell.get(str(cell.id))
                if layout_cell is None:
                    raise ValueError(f'Cell with ID {cell.id} not found in layout cells')

                cell.coords = layout_cell.coords
                cell.category = layout_cell.category


if __name__ == "__main__":
    main()

