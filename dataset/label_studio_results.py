import json
import logging
import os


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
