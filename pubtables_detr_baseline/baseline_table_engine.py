from enum import Enum
import os
import argparse
import subprocess
from tqdm import tqdm
import glob

import cv2
import numpy as np
from PIL import Image

from pubtables_1m.voc_xml import VocWord, VocLayout, VocObject, objects_intersect
from pero_ocr_word_engine.word_detection_engine import WordDetectionEngine, ImageContentType
from pero_ocr.core.layout import PageLayout
from organizer import utils
from table_transformer_baseline.table_structure_recognition_engine import TableStructureRecognitionEngine
from table_transformer_baseline.table_detection_engine import TableDetectionEngine

from organizer.tables.table_layout import TablePageLayout

class BaselineTableEngine:
    def __init__(self, work_dir: str, pero_script_path: str = None, image_content_type: ImageContentType = ImageContentType.czech_printed):
        self.work_dir = work_dir
        os.makedirs(self.work_dir, exist_ok=True)
        self.image_content_type = image_content_type

        self.word_detection_engine = WordDetectionEngine(work_dir, pero_script_path=pero_script_path,
                                                         image_content_type=image_content_type)
        self.table_detection_engine = TableDetectionEngine()
        self.table_structure_recognition_engine = TableStructureRecognitionEngine()

    def __call__(self) -> list[TablePageLayout]:
        return self.process_dir(self.work_dir)

    # add async version of process_dir so it can be used in fastapi request handler in a non-blocking way
    async def process_dir_async(self, work_dir: str) -> list[TablePageLayout]:
        return await utils.run_in_executor(self.process_dir, work_dir)

    def process_dir(self, work_dir: str) -> list[TablePageLayout]:
        table_layout_render_path = os.path.join(work_dir, 'table_layout_render')
        os.makedirs(table_layout_render_path, exist_ok=True)
        page_table_layout_render_path = os.path.join(work_dir, 'page_table_layout_render')
        os.makedirs(page_table_layout_render_path, exist_ok=True)
        table_layout_output_path = os.path.join(work_dir, 'table_layouts')
        os.makedirs(table_layout_output_path, exist_ok=True)
        page_table_layout_output_path = os.path.join(work_dir, 'page_table_layouts')
        os.makedirs(page_table_layout_output_path, exist_ok=True)
        table_crop_output_path = os.path.join(work_dir, 'table_crops')
        os.makedirs(table_crop_output_path, exist_ok=True)
        render_tsr_output_path = os.path.join(work_dir, 'rendered_tsr')
        os.makedirs(render_tsr_output_path, exist_ok=True)

        padding_around_table = 40

        page_layouts = self.word_detection_engine.process_dir(work_dir)
        page_layouts, image_files = self.get_layout_image_pairs(page_layouts, work_dir)
        table_page_layouts = []

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
            
            page_image_np = np.array(page_image) # table_crop)
            page_image_np = cv2.cvtColor(page_image_np, cv2.COLOR_RGB2BGR)

            print(f'found {len(table_crops)} tables in the image {image_file}')

            table_layouts_on_page = []

            for table_id, (table_crop, table_voc_object) in enumerate(zip(table_crops, table_voc_objects)):
                # use table structure recognition engine to recognize table structure in the image
                table_structure = self.table_structure_recognition_engine(table_crop)
                print(f'table_structure: {table_structure}')
                voc_objects = self.table_structure_recognition_engine.call_result_to_voc_objects(
                    table_structure) # , shift_coords=[table_voc_object.xmin, table_voc_object.ymin])
                print(f'voc_objects: {voc_objects}')

                voc_objects_on_page = []
                for obj in voc_objects:
                    # shift object coords to page coords
                    obj.xmin += table_voc_object.xmin - padding_around_table
                    obj.ymin += table_voc_object.ymin - padding_around_table
                    obj.xmax += table_voc_object.xmin - padding_around_table
                    obj.ymax += table_voc_object.ymin - padding_around_table
                    voc_objects_on_page.append(obj)

                # words_filtered = get_words_inside_table_crop(words.copy(), table_voc_object, padding=padding_around_table)

                # put everything to VocLayout object and call .to_table_layout
                voc_layout = VocLayout(objects=voc_objects_on_page,
                                        words=words,# words=words_filtered,
                                        width=table_crop.width, height=table_crop.height, depth=3,
                                        table_id=f'{page_layout.id}_table_{table_id}',
                                        tolerance=-8)

                rendered_tsr = voc_layout.render_tsr(pil_to_cv2(page_image))  # pil_to_cv2(table_crop))
                out_rendered_tsr_file = os.path.join(render_tsr_output_path, f"{os.path.basename(voc_layout.table_id)}.png")
                cv2.imwrite(out_rendered_tsr_file, rendered_tsr)

                table_layout = voc_layout.to_table_layout()
                if table_layout is None:
                    print(f"Table layout is None for table {table_id}. Skipping this table.")
                    continue
                table_layout_file_path = os.path.join(table_layout_output_path, f"{os.path.basename(table_layout.id)}.xml")
                print(f'Saving table layout to: {table_layout_file_path}')
                table_layout.to_table_pagexml(table_layout_file_path)
                print(f"Table layout:\n{table_layout}")

                rendered_image = table_layout.render_to_image(page_image_np)
                render_out_file = os.path.join(table_layout_render_path, f"{os.path.basename(table_layout.id)}.png")
                print(f'render_out_file: {render_out_file}')
                os.makedirs(os.path.dirname(render_out_file), exist_ok=True)
                cv2.imwrite(render_out_file, rendered_image)

                table_layouts_on_page.append(table_layout)

            # convert page_layout to table page layout
            table_page_layout = TablePageLayout.from_page_layout(page_layout)

            for table_layout in table_layouts_on_page:
                table_page_layout = add_table_to_table_page_layout(table_page_layout, table_layout)

            # render table page layout to image
            page_table_image = table_page_layout.render_to_image(page_image_np)
            page_table_image_file_path = os.path.join(page_table_layout_render_path, f"{os.path.basename(table_page_layout.id)}.png")
            print(f'Saving table page layout image to: {page_table_image_file_path}')
            cv2.imwrite(page_table_image_file_path, page_table_image)

            # save table page layout to file
            page_table_layout_file_path = os.path.join(page_table_layout_output_path, f"{os.path.basename(table_page_layout.id)}.xml")
            print(f'Saving table page layout to: {page_table_layout_file_path}')
            table_page_layout.to_table_pagexml(page_table_layout_file_path)

            table_page_layouts.append(table_page_layout)

        return table_page_layouts

    def get_layout_image_pairs(self, page_layouts: list[PageLayout], work_dir: str) -> tuple[list[PageLayout], list[str]]:
        approved_image_files = []
        approved_page_layouts = []

        for page_layout in page_layouts:
            image_path_regex = os.path.join(work_dir, f"{page_layout.id}.*")
            image_files = glob.glob(image_path_regex)

            if not image_files:
                print(f"No image files found for page layout {page_layout.id}. Skipping this page layout.")
                continue

            image_file = os.path.basename(image_files[0])
            # print(f"Using image file {image_file} for page layout {page_layout.id}")
            approved_image_files.append(image_file)  # Assuming the first match is the correct one
            approved_page_layouts.append(page_layout)

        return page_layouts, approved_image_files

def add_table_to_table_page_layout(table_page_layout: TablePageLayout, table_layout: TablePageLayout) -> TablePageLayout:
    """Add table layout to table page layout."""
    # Add table layout to table page layout
    if table_layout is None or table_layout.tables is None or len(table_layout.tables) == 0:
        print(f"Table layout is None or empty. Skipping this table layout.")
        return table_page_layout

    if (len(table_layout.tables) > 1):
        print(f"Table layout has more than 1 table. Saving only the first one.")

    table = table_layout.tables[0]
    table_page_layout.tables.append(table)

    # Remove everything from table page layout that intersects with the added table layout
    for region in table_page_layout.regions:
        # check if region intersects with the added table layout
        if not all(objects_intersect(region, table)):
            # if region does not intersect, let it be
            continue
        else:
            # if region intersect, remove lines that intersect with the added table layout
            lines_to_leave = []
            for line in region.lines:
                if not all(objects_intersect(table, line)):
                    lines_to_leave.append(line)

            region.lines = lines_to_leave

    # remove empty regions
    table_page_layout.regions = [region for region in table_page_layout.regions if len(region.lines) > 0]

    return table_page_layout

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
            word.xmax = min(padded_xmax, word.xmax - padded_xmin)
            word.ymax = min(padded_ymax, word.ymax - padded_ymin)
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
                      default=ImageContentType.czech_printed.value,
                      help="content type of the image")

    return args.parse_args()

def main():
    args = parse_args()

    table_engine = BaselineTableEngine(args.work_dir, image_content_type=args.content_type)
    table_page_layouts = table_engine()
    # word_detection_engine = WordDetectionEngine(args.work_dir, args.content_type)
    print(f"Loaded {len(table_page_layouts)} table page layouts: {table_page_layouts}")

if __name__ == "__main__":
    main()
