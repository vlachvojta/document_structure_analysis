from enum import Enum
import os
import argparse
import subprocess
from tqdm import tqdm

import cv2
import numpy as np
from PIL import Image

from pubtables_1m.voc_xml import VocWord
from pero_ocr.core.layout import PageLayout
from organizer import utils


class ImageContentType(Enum):
    czech_handwritten = 'czech_handwritten'
    czech_printed = 'czech_printed'

class WordDetectionEngine:
    """
    Word detection engine for OCR processing.
    This class is responsible for detecting words in images and providing the detected words in a page_layout as well as VocWord objects.
    """
    def __init__(self, work_dir: str, image_content_type: ImageContentType = ImageContentType.czech_handwritten,
                 pero_script_path: str = None):
        self.image_content_type = image_content_type

        self.work_dir = work_dir
        os.makedirs(self.work_dir, exist_ok=True)

        pero_script_default_path = os.path.join(os.path.dirname(__file__), 'run_pero_ocr.sh')
        self.pero_script_path = pero_script_path if pero_script_path else pero_script_default_path
        if not os.path.isfile(self.pero_script_path):
            raise FileNotFoundError(f"Pero OCR script not found at {self.pero_script_path}. Please provide a valid path")

        try:
            if isinstance(image_content_type, str):
                image_content_type = ImageContentType(image_content_type)
        except ValueError:
            raise ValueError(f"Invalid content type: {image_content_type}. Must be one of {[e.value for e in ImageContentType]}")

        config_name = f'pipeline_layout_sort_ocr_{image_content_type.value}.ini'
        self.config_file = os.path.join(os.path.dirname(self.pero_script_path), config_name)
        if not os.path.isfile(self.config_file):
            raise FileNotFoundError(f"Pero OCR config file not found at {self.config_file}. Please provide a valid path")

    def __call__(self) -> list[PageLayout]:
        return self.process_dir(self.work_dir)

    def process_dir(self, work_dir: str) -> list[PageLayout]:
        """Process the input image and return a PageLayout object with detected words."""
        command = [f'{self.pero_script_path} {work_dir} {self.config_file}']
        print(f"Running pero_ocr script: {' '.join(command)}")
        # try:
        pero_ocr_result = subprocess.run(command, shell=True) #, check=True)
        # except subprocess.CalledProcessError as e:
        #     print(f"Error running pero_ocr script: {e}")
        #     return []

        # Check if the script executed successfully
        if not pero_ocr_result.returncode == 0:
            print(f"Script failed with return code {pero_ocr_result.returncode}")
            print(f"Error: {pero_ocr_result.stderr}")
            return []

        xml_output_path = os.path.join(work_dir, "xml")
        xml_files = [f for f in os.listdir(xml_output_path) if f.endswith('.xml')]

        page_layouts = []
        for xml_file in tqdm(xml_files, desc='Loading page layouts from xml files'):
            xml_file_path = os.path.join(xml_output_path, xml_file)
            page_layout = PageLayout(id=xml_file, file=xml_file_path)
            page_layouts.append(page_layout)

        return page_layouts

    def process_image(self, image: str | Image.Image | np.ndarray) -> PageLayout:
        """Save image to work_dir and then self.__call__() to get words in page_layout."""
        # str is path to the image, take dirname and pass it to pero_ocr
        # Image.Image is image object, save it to work_dir and pass it to pero_ocr
        # np.ndarray is image object, save it to work_dir and pass it to pero_ocr
        ...

    def extract_words(self, page_layout: PageLayout) -> list[VocWord]:
        """
        Extract words from the page layout and return them as VocWord objects.
        """
        voc_words = []
        for word in page_layout.words_iterator():
            l, t, r, b = utils.polygon_to_ltrb(word.polygon)

            # Create a VocWord object from the word data
            voc_word = VocWord(
                xmin=float(l),
                ymin=float(t),
                xmax=float(r),
                ymax=float(b),
                text=word.transcription,
                transcription_confidence=word.transcription_confidence,
            )
            voc_words.append(voc_word)
        return voc_words

def parse_args():
    args = argparse.ArgumentParser()
    # args.add_argument("--model", type=str, default="microsoft/table-transformer-detection")
    # args.add_argument("--device", type=str, default="cuda")
    args.add_argument("-i", "--work-dir", type=str, default="example_data")
    args.add_argument("-c", "--content-type", type=str, choices=[e.value for e in ImageContentType],
                      default=ImageContentType.czech_handwritten.value,
                      help="content type of the image")

    return args.parse_args()

def main():
    args = parse_args()

    word_detection_engine = WordDetectionEngine(args.work_dir, args.content_type)
    page_layouts = word_detection_engine()
    print(f"Loaded {len(page_layouts)} page layouts: {page_layouts}")

    # extract words from the first page layout
    if page_layouts:
        words = word_detection_engine.extract_words(page_layouts[0])
        word_len = len(words)
        print(f"Extracted {word_len} words from the first page layout, such as:")
        print(words[:min(10, word_len)])
    else:
        print("No page layouts found. Please check the input directory.")

if __name__ == "__main__":
    main()
