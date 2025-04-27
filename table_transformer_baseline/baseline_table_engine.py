from enum import Enum
import os
import argparse
import subprocess
from tqdm import tqdm
import glob

import cv2
import numpy as np
from PIL import Image

from pubtables_1m.voc_xml import VocWord, VocLayout, VocObject
from pero_ocr_word_engine.word_detection_engine import WordDetectionEngine, ImageContentType
from pero_ocr.core.layout import PageLayout
from organizer import utils
from table_transformer_baseline.table_structure_recognition_engine import TableStructureRecognitionEngine
from table_transformer_baseline.table_detection_engine import TableDetectionEngine

from organizer.tables.table_layout import TablePageLayout

class TableEngine:
    def __init__(self, work_dir: str, pero_script_path: str = None, image_content_type: ImageContentType = ImageContentType.czech_handwritten):
        self.work_dir = work_dir
        os.makedirs(self.work_dir, exist_ok=True)
        self.image_content_type = image_content_type

        self.word_detection_engine = WordDetectionEngine(work_dir, pero_script_path=pero_script_path,
                                                         image_content_type=image_content_type)
        self.table_detection_engine = TableDetectionEngine()
        self.table_structure_recognition_engine = TableStructureRecognitionEngine()

    def __call__(self) -> list[TablePageLayout]:
        return self.process_dir(self.work_dir)

    def process_dir(self, work_dir: str) -> list[TablePageLayout]:
        table_layout_render_path = os.path.join(work_dir, 'table_layout_render')
        os.makedirs(table_layout_render_path, exist_ok=True)
        table_layout_output_path = os.path.join(work_dir, 'table_layouts')
        os.makedirs(table_layout_output_path, exist_ok=True)
        table_crop_output_path = os.path.join(work_dir, 'table_crops')
        os.makedirs(table_crop_output_path, exist_ok=True)
        render_tsr_output_path = os.path.join(work_dir, 'rendered_tsr')
        os.makedirs(render_tsr_output_path, exist_ok=True)

        padding_around_table = 40

        page_layouts = self.word_detection_engine.process_dir(work_dir)
        page_layouts, image_files = self.get_layout_image_pairs(page_layouts, work_dir)

        print(f'Found and processed {len(page_layouts)} page layouts in {work_dir}')

        print(f'page_layouts: {page_layouts}')
        print(f'image_files: {image_files}')

        for page_layout, image_file in zip(page_layouts, image_files):
            print(f"\n\nProcessing image: {image_file}")
            image_path = os.path.join(work_dir, image_file)

            # image_path = os.path.join(work_dir, image_file)
            page_image = Image.open(image_path).convert("RGB")

            words = self.word_detection_engine.extract_words(page_layout)

            # use table detection engine to detect tables in the image
            table_detection_results = self.table_detection_engine(page_image)
            table_voc_objects = self.table_detection_engine.get_tables_as_voc_objects(table_detection_results)
            table_crops = self.table_detection_engine.get_table_crops(page_image, table_detection_results, padding=padding_around_table)

            for i, table_crop in enumerate(table_crops):
                table_crop_out_file = os.path.join(table_crop_output_path, f"{os.path.splitext(image_file)[0]}_crop_{i}.png")
                table_crop.save(table_crop_out_file)
                print(f"Saved crop to {table_crop_out_file}")

            print(f'found {len(table_crops)} tables in the image {image_file}')

            for table_id, (table_crop, table_voc_object) in enumerate(zip(table_crops, table_voc_objects)):
                # use table structure recognition engine to recognize table structure in the image
                table_structure = self.table_structure_recognition_engine(table_crop)
                print(f'table_structure: {table_structure}')
                voc_objects = self.table_structure_recognition_engine.call_result_to_voc_objects(
                    table_structure) # , shift_coords=[table_voc_object.xmin, table_voc_object.ymin])
                print(f'voc_objects: {voc_objects}')

                words_filtered = get_words_inside_table_crop(words.copy(), table_voc_object, padding=padding_around_table)

                # put everything to VocLayout object and call .to_table_layout
                voc_layout = VocLayout(objects=voc_objects,
                                        words=words_filtered,
                                        width=table_crop.width, height=table_crop.height, depth=3,
                                        table_id=f'{page_layout.id}_table_{table_id}',
                                        tolerance=-10)
                                        # xml_file=os.path.splitext(os.path.basename(image_path))[0],
                                        # word_file=os.path.splitext(os.path.basename(image_path))[0])

                rendered_tsr = voc_layout.render_tsr(pil_to_cv2(table_crop))
                out_rendered_tsr_file = os.path.join(render_tsr_output_path, f"{os.path.basename(voc_layout.table_id)}.png")
                cv2.imwrite(out_rendered_tsr_file, rendered_tsr)

                table_layout = voc_layout.to_table_layout()
                table_layout_file_path = os.path.join(table_layout_output_path, f"{os.path.basename(table_layout.id)}.xml")
                print(f'Saving table layout to: {table_layout_file_path}')
                table_layout.to_table_pagexml(table_layout_file_path)
                print(f"Table layout:\n{table_layout}")
                # render table layout to image
                image_np = np.array(table_crop)
                image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
                # print(f"image_np shape: {image_np.shape}")

                # save image to ./test.png file
                cv2.imwrite('./test.png', image_np)

                # show image_np for debugging
                # cv2.imshow('image', image_np)
                # cv2.waitKey(1)
                # cv2.destroyAllWindows()

                rendered_image = table_layout.render_to_image(image_np.copy())
                # print(f"rendered_image shape: {rendered_image.shape}")
                # print(rendered_image)
                # show image_np for debugging
                render_out_file = os.path.join(table_layout_render_path, f"{os.path.basename(table_layout.id)}.png")
                print(f'render_out_file: {render_out_file}')
                os.makedirs(os.path.dirname(render_out_file), exist_ok=True)
                cv2.imwrite(render_out_file + '-orig.png', image_np)
                cv2.imwrite(render_out_file, rendered_image)


                # locate table page layout in the whole page layout

        # table_detection_results = self.table_detection_engine(image)

        # table_page_layouts = []

        # for table in table_detection_results:
        #     table_img = None# cut out table from image using table bbox
        #     table_structure = self.table_structure_recognition_engine(table_img)
        #     words = self.word_detection_engine(table_img)

        #     table_page_layout = self.create_table_page_layout(table_structure, words)
        #     table_page_layouts.append(table_page_layout)

        return [table_layout]

    def get_layout_image_pairs(self, page_layouts: list[PageLayout], work_dir: str) -> tuple[list[PageLayout], list[str]]:
        # print(f'\n\nLooking for images in {work_dir}')

        approved_image_files = []
        approved_page_layouts = []

        for page_layout in page_layouts:
            image_path_regex = os.path.join(work_dir, f"{page_layout.id}.*")
            image_files = glob.glob(image_path_regex)

            # print(f"for page layout {page_layout.id} and regex {image_path_regex} "
            #       f"found {len(image_files)} images: {image_files}")

            if not image_files:
                print(f"No image files found for page layout {page_layout.id}. Skipping this page layout.")
                continue

            image_file = os.path.basename(image_files[0])
            # print(f"Using image file {image_file} for page layout {page_layout.id}")
            approved_image_files.append(image_file)  # Assuming the first match is the correct one
            approved_page_layouts.append(page_layout)

        # print(f'returning images: {approved_image_files}')

        return page_layouts, approved_image_files

def get_words_inside_table_crop(words: list[VocWord], table: VocObject, padding: int = 0) -> list[VocWord]:
    """Get words inside the table crop and positions them to table coords."""
    words_inside_table = []
    for word in words:

        padded_xmin = table.xmin - padding
        padded_ymin = table.ymin - padding
        padded_xmax = table.xmax + padding
        padded_ymax = table.ymax + padding

        if word_is_inside_object(word, table, padding=padding):
            # shift word coords to table coords
            word.xmin = max(0, word.xmin - padded_xmin)
            word.ymin = max(0, word.ymin - padded_ymin)
            word.xmax = min(table.xmax, word.xmax - padded_xmin)
            word.ymax = min(table.ymax, word.ymax - padded_ymin)
            words_inside_table.append(word)

    return words_inside_table

def word_is_inside_object(word: VocWord, obj: VocObject, padding: int = 0) -> bool:
    """Check if word is inside the object."""
    padded_xmin = obj.xmin - padding
    padded_ymin = obj.ymin - padding
    padded_xmax = obj.xmax + padding
    padded_ymax = obj.ymax + padding

    if word.xmax >= padded_xmin and word.xmin <= padded_xmax and word.ymax >= padded_ymin and word.ymin <= padded_ymax:
        return True
    return False

def pil_to_cv2(image: Image.Image) -> np.ndarray:
    """Convert a PIL image to a CV2 image (numpy array)."""
    image_np = np.array(image)
    image_bgr = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
    return image_bgr

def parse_args():
    args = argparse.ArgumentParser()
    # args.add_argument("--model", type=str, default="microsoft/table-transformer-detection")
    # args.add_argument("--device", type=str, default="cuda")
    args.add_argument("-i", "--work-dir", type=str, default="example_data/whole_pipeline_test/")
    args.add_argument("-c", "--content-type", type=str, choices=[e.value for e in ImageContentType],
                      default=ImageContentType.czech_handwritten.value,
                      help="content type of the image")

    return args.parse_args()

def main():
    args = parse_args()


    table_engine = TableEngine(args.work_dir, image_content_type=args.content_type)
    table_page_layouts = table_engine()
    # word_detection_engine = WordDetectionEngine(args.work_dir, args.content_type)
    print(f"Loaded {len(table_page_layouts)} table page layouts: {table_page_layouts}")

    # extract words from the first page layout
    # if table_page_layouts:
    #     words = word_detection_engine.extract_words(table_page_layouts[0])
    #     word_len = len(words)
    #     print(f"Extracted {word_len} words from the first page layout, such as:")
    #     print(words[:min(10, word_len)])
    # else:
    #     print("No page layouts found. Please check the input directory.")

if __name__ == "__main__":
    main()
