import lxml.etree as ET
import json
from enum import Enum
import re
import os

import numpy as np
import cv2

from organizer.tables.order_guessing import cluster_objects
from organizer.tables.table_layout import TablePageLayout, TableRegion, TableCell
from organizer import utils
from pero_ocr.core.layout import TextLine, Word


class VocWord:
    def __init__(self, word: dict):
        # self.bbox = word['bbox']
        self.xmin, self.ymin, self.xmax, self.ymax = word['bbox']
        # self.bbox = (self.xmin, self.ymin, self.xmax, self.ymax)

        self.text = word['text']
        self.flags = word['flags']
        self.span_num = word['span_num']
        self.line_num = word['line_num']
        self.block_num = word['block_num']

    def ltrb(self):
        return self.xmin, self.ymin, self.xmax, self.ymax


class VocObject:
    def __init__(self, obj: ET.Element):
        try:
            self.category = ObjectCategory(obj.find('name').text)
        except ValueError:
            print(f'Warning: Unknown object category: {obj.find("name").text}')
            self.category = obj.find('name').text

        self.pose = obj.find('pose').text
        self.truncated = int(obj.find('truncated').text)
        self.difficult = int(obj.find('difficult').text)
        self.occluded = int(obj.find('occluded').text)

        self.bndbox = obj.find('bndbox')
        self.xmin = float(self.bndbox.find('xmin').text)
        self.ymin = float(self.bndbox.find('ymin').text)
        self.xmax = float(self.bndbox.find('xmax').text)
        self.ymax = float(self.bndbox.find('ymax').text)

    def ltrb(self):
        return self.xmin, self.ymin, self.xmax, self.ymax

    def lt(self):
        return self.xmin, self.ymin
    
    def rb(self):
        return self.xmax, self.ymax


class ObjectCategory(Enum):
    table = 'table'
    table_column = 'table column'
    table_row = 'table row'
    table_projected_row_header = 'table projected row header'
    table_spanning_cell = 'table spanning cell'
    table_column_header = 'table column header'

category_colors = {
    ObjectCategory.table: (255, 0, 0), # blue
    ObjectCategory.table_column: (240, 240, 70), # cyan
    ObjectCategory.table_row: (128, 128, 0), # teal
    ObjectCategory.table_projected_row_header: (0, 0, 128),  # maroon
    ObjectCategory.table_spanning_cell: (0, 0, 128),  # maroon
}


class VocLayout:
    grid_categories = [ObjectCategory.table_column,
                       ObjectCategory.table_row]
    joind_cell_categories = [ObjectCategory.table_projected_row_header,
                             ObjectCategory.table_spanning_cell]

    def __init__(self, xml_file: str, word_file: str = None):
        self.xml_file = xml_file
        self.word_file = word_file
        self.table_id = xml_file.replace('.xml', '')

        self.load_voc_xml(xml_file)

        if word_file is not None:
            self.load_words(word_file)

        self.warnings_sent = []

    def load_voc_xml(self, xml_file: str):
        try:
            self.tree = ET.parse(xml_file)
            self.root = self.tree.getroot()
        except Exception as e:
            print(f'Error loading xml file: {xml_file}')
            print(e)
            return None

        size = self.root.find('size')
        self.width = int(size.find('width').text)
        self.height = int(size.find('height').text)
        self.depth = int(size.find('depth').text)

        objects = self.root.findall('object')
        self.objects = [VocObject(obj) for obj in objects]

    def load_words(self, word_file: str):
        with open(word_file) as f:
            words = json.load(f)

        self.words = [VocWord(word) for word in words]

    def get_objects(self, categories: list[ObjectCategory] = None) -> list[VocObject]:
        selected_categories = []

        for obj in self.objects:
            if not categories or obj.category in categories:
                selected_categories.append(obj)

        return selected_categories

    def render_tsr(self, image: np.ndarray) -> np.ndarray:
        # render words
        for word in self.words:
            l, t, r, b = word.ltrb()
            word_color_pink = (255, 190, 220)
            cv2.rectangle(image, (int(l), int(t)), (int(r), int(b)), word_color_pink, 1)
            image_words = image.copy()

        grid_objects = self.get_objects(self.grid_categories)
        joind_cells_objects = self.get_objects(self.joind_cell_categories)

        # render grid objects
        for obj in grid_objects:
            l, t, r, b = obj.ltrb()
            cv2.rectangle(image, (int(l), int(t)), (int(r), int(b)), category_colors[obj.category], 1)

        # render joined cells (on top of grid objects)
        for obj in joind_cells_objects:
            l, t, r, b = obj.ltrb()
            # copy part of the image and place it on top of the image (to be on top already existing borders)
            joined_cell_image = image_words[int(t):int(b), int(l):int(r)].copy()
            image[int(t):int(b), int(l):int(r)] = joined_cell_image

            cv2.rectangle(image, (int(l), int(t)), (int(r), int(b)), category_colors[obj.category], 1)

        # re-render all table objects to be on top
        table_objects = [obj for obj in self.objects if obj.category == ObjectCategory.table]
        for obj in table_objects:
            l, t, r, b = obj.ltrb()
            cv2.rectangle(image, (int(l), int(t)), (int(r), int(b)), category_colors[obj.category], 1)

        return image

    def to_table_layout(self) -> TablePageLayout:
        rows = [obj for obj in self.objects if obj.category == ObjectCategory.table_row]
        rows.sort(key=lambda x: x.ymin)

        columns = [obj for obj in self.objects if obj.category == ObjectCategory.table_column]
        columns.sort(key=lambda x: x.xmin)

        joined_cells = self.get_joined_cells(rows, columns)
        cells, cells_structure = self.create_cells(rows, columns, joined_cells)
        cells_with_words = self.assign_words_to_cells(rows, columns, cells, cells_structure, joined_cells, table_id=self.table_id)
        for cell in cells_with_words:
            if cell is not None:
                self.order_words_in_cell(cell, table_id=self.table_id)

        # check only one table region in the file
        table_objects = self.get_objects([ObjectCategory.table])
        if len(table_objects) != 1:
            print(f'Warning: Expected exactly one table region, found {len(table_objects)}, skipping file {self.xml_file}')
            self.warnings_sent.append(f'more than one table region')
            return None

        # create table region
        table_polygon = utils.ltrb_to_polygon(*table_objects[0].ltrb())
        table_region = TableRegion(id=os.path.basename(self.table_id), coords=table_polygon)
        table_region.insert_cells([cell for cell in cells_with_words if cell is not None])

        # create page layout
        page_layout = TablePageLayout(id=self.table_id, file=self.xml_file, page_size=(self.height, self.width))
        page_layout.tables.append(table_region)

        return page_layout

    def get_joined_cells(self, rows: list[VocObject], columns: list[VocObject]) -> list[TableCell]:
        """Find joined cells and assign them position (left-most column, top-most row) and span."""
        tolerance = 3
        get_joined_cells = [obj for obj in self.objects if obj.category in self.joind_cell_categories]

        joined_cells = []

        for cell_id, cell in enumerate(get_joined_cells):
            intersecting_rows = [
                row for row in rows
                if cell.ymin < row.ymax - tolerance and cell.ymax > row.ymin + tolerance]

            intersecting_columns = [
                column for column in columns 
                if cell.xmin < column.xmax - tolerance and cell.xmax > column.xmin + tolerance]

            if len(intersecting_rows) == 0 or len(intersecting_columns) == 0:
                print(f'Warning: Joined cell {cell} does not intersect any row or column')
                self.warnings_sent.append(f'joined cell does not intersect any row or column')
                continue

            # find the index of left-most column and top-most row
            left_column_idx = columns.index(intersecting_columns[0])

            top_row_idx = rows.index(intersecting_rows[0])

            # find the span of the cell
            column_span = len(intersecting_columns)
            row_span = len(intersecting_rows)

            polygon = utils.ltrb_to_polygon(*cell.ltrb())

            new_table_cell = TableCell(str(cell_id), polygon,
                                       category=cell.category.value, row=top_row_idx, col=left_column_idx,
                                       row_span=row_span, col_span=column_span)
            joined_cells.append(new_table_cell)


        return joined_cells

    def create_cells(self, rows: list[VocObject], columns: list[VocObject], joined_cells: list[TableCell]) -> tuple[list[TableCell], np.ndarray[int]]:
        # create just table structure of non joined cells
        EMPTY_CELL_ID = -42
        cells = []
        cells_structure = np.zeros((len(rows), len(columns)), dtype=int) + EMPTY_CELL_ID
        # cells_np = np.empty((max_row, max_col), dtype=TableCell)

        for row_idx, row in enumerate(rows):
            for col_idx, column in enumerate(columns):
                polygon = utils.ltrb_to_polygon(column.xmin, row.ymin, column.xmax, row.ymax)
                cell_id = len(cells)
                new_cell = TableCell(id=f'c{cell_id:03d}', coords=polygon, row=row_idx, col=col_idx, row_span=1, col_span=1)
                cells.append(new_cell)
                cells_structure[row_idx, col_idx] = cell_id

        # join cells according to the joined_cells list
        for joined_cell in joined_cells:
            row = joined_cell.row
            col = joined_cell.col
            row_span = joined_cell.row_span
            col_span = joined_cell.col_span

            delete_cells = cells_structure[row:row + row_span, col:col + col_span].flatten()
            for cell_id in delete_cells:
                cells[cell_id] = None

            cell_id = len(cells)
            cells_structure[row:row + row_span, col:col + col_span] = cell_id
            joined_cell.id = str(cell_id)
            cells.append(joined_cell)

        return cells, cells_structure

    def assign_words_to_cells(self, rows: list[VocObject], columns: list[VocObject], cells: list[TableCell], cells_structure: np.ndarray[int], joined_cells: list[TableCell],
                              table_id: str) -> list[TableCell]:
        """Assign words to cells based on their position. Words are added to a single TextLine in the order they were in the JSON file."""
        tolerance = 3

        for word in self.words:
            # filter out words that are not inside the table
            if (word.xmax < columns[0].xmin or word.xmin > columns[-1].xmax or
                word.ymax < rows[0].ymin or word.ymin > rows[-1].ymax):
                continue

            # find the row and column of the word
            intersecting_rows = [
                row for row in rows
                if word.ymin < row.ymax - tolerance and word.ymax > row.ymin + tolerance]

            intersecting_columns = [
                column for column in columns
                if word.xmin < column.xmax and word.xmax > column.xmin]
                # if word.xmin < column.xmax - tolerance and word.xmax > column.xmin + tolerance]

            if len(intersecting_rows) == 0 or len(intersecting_columns) == 0:
                if len(intersecting_rows) == 0 and len(intersecting_columns) == 0:
                    print(f'Warning({table_id}): Word {word} ({word.text} at {word.ltrb()}) does not intersect any row or column')
                    self.warnings_sent.append(f'word does not intersect any row or column')
                elif len(intersecting_rows) == 0:
                    print(f'Warning({table_id}): Word {word} ({word.text} at {word.ltrb()}) does not intersect any row')
                    self.warnings_sent.append(f'word does not intersect any row')
                elif len(intersecting_columns) == 0:
                    print(f'Warning({table_id}): Word {word} ({word.text} at {word.ltrb()}) does not intersect any column')
                    self.warnings_sent.append(f'word does not intersect any column')
                continue
            elif len(intersecting_rows) > 1 or len(intersecting_columns) > 1:
                # find the joined cell that the word intersects (cell interestcs more than one row or column)
                intersecting_joined_cells = []
                for cell in joined_cells:
                    cell_l, cell_t, cell_r, cell_b = utils.polygon_to_ltrb(cell.coords)
                    if word.xmin < cell_r - tolerance and word.xmax > cell_l + tolerance and word.ymin < cell_b - tolerance and word.ymax > cell_t + tolerance:
                        intersecting_joined_cells.append(cell)

                if len(intersecting_joined_cells) == 1:
                    cell = intersecting_joined_cells[0]
                    row_idx = cell.row
                    col_idx = cell.col
                elif len(intersecting_joined_cells) == 0:
                    print(f'Warning({table_id}): Word {word} ({word.text} at {word.ltrb()}) does not intersect any rows, columns or joined cells')
                    self.warnings_sent.append(f'word does not intersect any rows, columns or joined cells')
                    continue
                elif len(intersecting_joined_cells) > 1:
                    print(f'Warning({table_id}): Word {word} ({word.text} at {word.ltrb()}) intersects more than one joined cell')
                    self.warnings_sent.append(f'word intersects more than one joined cell')
                    continue
            else:  # word intersects exactly one row and one column
                row_idx = rows.index(intersecting_rows[0])
                col_idx = columns.index(intersecting_columns[0])

            cell_id = cells_structure[row_idx, col_idx]
            cell = cells[cell_id]

            # word to textline
            word_polygon = utils.ltrb_to_polygon(*word.ltrb())
            word = Word(id=str(word.span_num), polygon=word_polygon, transcription=word.text)

            if cell.lines:
                cell.lines[0].words.append(word)
            else:
                textline = TextLine(id=f'{cell.id}_l000', polygon=cell.coords, transcription='')
                textline.words.append(word)
                cell.lines.append(textline)
        return cells

    def order_words_in_cell(self, cell: TableCell, table_id: str) -> TableCell:
        if cell is None or len(cell.lines) == 0:
            return cell

        if len(cell.lines) > 1:
            print(f'Warning({table_id}): Cell {cell.id} has more than one line ({len(cell.lines)}) in table {self.table_id}. Getting words from all to reorder them.')
            self.warnings_sent.append(f'more than one line in cell')

        words = [word for line in cell.lines for word in line.words]

        eps = 1
        word_id_clusters = cluster_objects(words, min_samples=1, eps=eps)
        overlap_indices = self.check_word_overlap(words, word_id_clusters)
        if len(overlap_indices) > 0 and len(word_id_clusters) == 1:
            print(f'Warning({table_id}): Overlapping words in cell {cell.id}, but only one cluster found. Consider adjusting eps parameter (currently {eps}).')
            print(f'\tOverlapping words: {overlap_indices}')
            print(f'\tClusters: {word_id_clusters}')
            self.warnings_sent.append(f'overlapping words in cell')

        # create new textline from each cluster of words
        textlines = []
        for cluster_id, cluster in enumerate(word_id_clusters):
            words_in_cluster = [words[word_id] for word_id in cluster]

            textline_polygon = polygon_around_polygons([word.polygon for word in words_in_cluster])
            transcription = ' '.join([word.transcription for word in words_in_cluster])
            textline = TextLine(id=f'{cell.id}_l{cluster_id:03d}', polygon=textline_polygon, transcription=transcription)
            textline.words = words_in_cluster
            textlines.append(textline)

        cell.lines = textlines
        # make cell coords the bounding box around all textlines
        # cell.coords = polygon_around_polygons([textline.polygon for textline in textlines])

        return cell

    def check_word_overlap(self, words: list[Word], word_id_clusters: list[list[int]]) -> np.ndarray:
        tolerance = 3
        words = sorted(words, key=lambda x: np.min(x.polygon[:, 0])) # sort by xmin coordinate

        # create bounding boxes consisting of xmin, xmax coordinates
        bboxes = []
        for word in words:
            bboxes.append([np.min(word.polygon[:, 0]), np.max(word.polygon[:, 0])])
        bboxes = np.array(bboxes)

        # compare xmins of words with xmaxs of previous words, resulting in bool array of same length as bboxes
        overlap_bool = bboxes[1:, 0] < bboxes[:-1, 1] - tolerance
        # get indices of true values
        overlap_indices = np.where(overlap_bool)[0]

        return overlap_indices

def polygon_around_polygons(polygons: list[np.ndarray]) -> np.ndarray:
    l = min([np.min(polygon[:, 0]) for polygon in polygons])
    t = min([np.min(polygon[:, 1]) for polygon in polygons])
    r = max([np.max(polygon[:, 0]) for polygon in polygons])
    b = max([np.max(polygon[:, 1]) for polygon in polygons])

    return utils.ltrb_to_polygon(l, t, r, b)

# words JSON file example:
# [
#   {
#     "bbox": [
#       31.47979797979798,
#       0.9930303030303094,
#       95.38327020202019,
#       12.852196969696962
#     ],
#     "text": "Comparisons",
#     "flags": 0,
#     "span_num": 515,
#     "line_num": 0,
#     "block_num": 0
#   },
#   {
#     "bbox": [
#       98.13597222222222,
#       0.9930303030303094,
#       108.02532828282827,
#       12.852196969696962
#     ],
#     "text": "of",
#     "flags": 0,
#     "span_num": 516,
#     "line_num": 0,
#     "block_num": 0
#   },
#   {


# VOC XML file example:
# <?xml version="1.0" ?>
# <annotation>
#    <folder/>
#    <filename>PMC1064076_table_0.jpg</filename>
#    <path>PMC1064076_table_0.jpg</path>
#    <source>
#       <database>PubTables1M-Structure</database>
#    </source>
#    <size>
#       <width>685</width>
#       <height>323</height>
#       <depth>3</depth>
#    </size>
#    <segmented>0</segmented>
#    <object>
#       <name>table</name>
#       <pose>Frontal</pose>
#       <truncated>0</truncated>
#       <difficult>0</difficult>
#       <occluded>0</occluded>
#       <bndbox>
#          <xmin>36.5556</xmin>
#          <ymin>36.6549</ymin>
#          <xmax>646.0124</xmax>
#          <ymax>284.3667</ymax>
#       </bndbox>
#    </object>
#    <object>
#       <name>table projected row header</name>
#       <pose>Frontal</pose>
#       <truncated>0</truncated>
#       <difficult>0</difficult>
#       <occluded>0</occluded>
#       <bndbox>
#          <xmin>36.5556</xmin>
#          <ymin>52.8597</ymin>
#          <xmax>646.0124</xmax>
#          <ymax>72.8222</ymax>
#       </bndbox>
#    </object>
#    ...
# </annotation>
