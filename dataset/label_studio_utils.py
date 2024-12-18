import json
import logging
import os

import numpy as np


class LabelStudioResults:
    def __init__(self, label_file: str, verbose: bool = False):
        self.label_file = label_file

        if verbose:
            logging.basicConfig(level=logging.DEBUG, format='[%(levelname)-s]\t- %(message)s')
        else:
            logging.basicConfig(level=logging.INFO,format='[%(levelname)-s]\t- %(message)s')

        self.logger = logging.getLogger(__name__)

        with open(label_file, 'r') as f:
            self.data = json.load(f)

        self.logger.info(f'{len(self.data)} tasks loaded from {self.label_file}')
        # self.logger.debug('\n' + json.dumps(self.data[0], indent=4))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx: int):
        return self.data[idx]

    def get_images(self, base_name=False) -> list[str]:
        images = [task['data']['image'] for task in self.data]
        if base_name:
            return [os.path.basename(image) for image in images]
        return images

    def filter_tasks_using_images(self, images: list[str]):
        # delete tasks that don't have corresponding images
        if len(images) == 0:
            raise ValueError('No images to filter')

        self.data = [task for task in self.data if os.path.basename(task['data']['image']) in images]

    def replace_image_names(self, old_name: str, new_name: str, base_name: bool = False):
        for task in self.data:
            task['data']['image'] = task['data']['image'].replace(old_name, new_name)
            if base_name:
                task['data']['image'] = os.path.basename(task['data']['image'])


def label_studio_coords_to_xywh(coords: dict, image_shape: list[int], padding: int = 0) -> tuple[int, int, int, int]:

    assert 'x' in coords and 'y' in coords and 'width' in coords and 'height' in coords, \
        f'Invalid coordinates: {coords}. Expected keys: x, y, width, height'

    x, y = coords['x'], coords['y']
    w, h = coords['width'], coords['height']

    assert len(image_shape) == 2, f'Invalid image shape: {image_shape}. Expected 2D shape.'

    # x, y, w, h = coords
    x = int(x / 100 * image_shape[1])
    y = int(y / 100 * image_shape[0])
    w = int(w / 100 * image_shape[1])
    h = int(h / 100 * image_shape[0])

    x, y, w, h = add_padding([x, y, w, h], padding, image_shape)

    return x, y, w, h

def add_padding(coords: list[int], padding: int, image_shape: list[int]) -> list[int]:
    x, y, w, h = coords

    x1 = max(0, x - padding)
    y1 = max(0, y - padding)
    x2 = min(image_shape[1], x + w + padding)
    y2 = min(image_shape[0], y + h + padding)

    return [x1, y1, x2 - x1, y2 - y1]

# x = max(0, x - self.padding) 
# y = max(0, y - self.padding)
# w = min(img.shape[1], w + 2 * self.padding)
# h = min(img.shape[0], h + 2 * self.padding)

def get_label_studio_coords(bbox, img_shape):
    left_top = bbox[0]
    right_bottom = bbox[2]

    width = right_bottom[0] - left_top[0]
    height = right_bottom[1] - left_top[1]
    x = left_top[0]
    y = left_top[1]

    width /= img_shape[1]
    height /= img_shape[0]
    x /= img_shape[1]
    y /= img_shape[0]

    width *= 100
    height *= 100
    x *= 100
    y *= 100

    return x, y, width, height

def get_label_studio_coords(polygon: np.ndarray, img_shape):
    # polygon is a 2d list of points (x, y)
    x = np.min(polygon[:, 0])
    y = np.min(polygon[:, 1])
    width = np.max(polygon[:, 0]) - x
    height = np.max(polygon[:, 1]) - y

    width /= img_shape[1]
    height /= img_shape[0]
    x /= img_shape[1]
    y /= img_shape[0]

    width *= 100
    height *= 100
    x *= 100
    y *= 100

    return x, y, width, height
