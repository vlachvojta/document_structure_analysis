import lxml.etree as ET
import json
from enum import Enum

import numpy as np
import cv2

from organizer.tables.table_layout import TablePageLayout, TableRegion, TableCell


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

        self.load_voc_xml(xml_file)

        if word_file is not None:
            self.load_words(word_file)

    def load_voc_xml(self, xml_file: str):
        try:
            self.tree = ET.parse(xml_file)
            self.root = self.tree.getroot()
        except Exception as e:
            print(f'Error loading xml file: {xml_file}')
            print(e)
            return None

        self.size = self.root.find('size')
        self.width = int(self.size.find('width').text)
        self.height = int(self.size.find('height').text)
        self.depth = int(self.size.find('depth').text)

        objects = self.root.findall('object')
        self.objects = [VocObject(obj) for obj in objects]

    def load_words(self, word_file: str):
        with open(word_file) as f:
            words = json.load(f)

        self.words = [VocWord(word) for word in words]

    def render_tsr(self, image: np.ndarray) -> np.ndarray:
        # render words
        for word in self.words:
            l, t, r, b = word.ltrb()
            word_color_pink = (255, 190, 220)
            cv2.rectangle(image, (int(l), int(t)), (int(r), int(b)), word_color_pink, 1)
            image_words = image.copy()

        grid_objects = [obj for obj in self.objects
                        if obj.category in self.grid_categories]
        joind_cells_objects = [obj for obj in self.objects 
                               if obj.category in self.joind_cell_categories]

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
