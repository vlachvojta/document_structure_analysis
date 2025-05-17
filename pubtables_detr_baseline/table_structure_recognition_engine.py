import os
# import sys
import argparse

import cv2
import numpy as np

# from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import DetrImageProcessor, TableTransformerForObjectDetection
import torch

from pubtables_1m.voc_xml import VocObject, VocLayout

class TableStructureRecognitionEngine:
    """Use microsoft table structure recognition model to recognize table structure in images."""

    def __init__(self, model_name: str = "microsoft/table-transformer-structure-recognition",
                 device: str = "cuda"):
        self.device = device
        self.feature_extractor = DetrImageProcessor()
        self.tsr_model = TableTransformerForObjectDetection.from_pretrained(model_name)

        self.id2label = self.tsr_model.config.id2label

    def __call__(self, image) -> list[list[float]]:
        encoding = self.feature_extractor(image, return_tensors="pt")

        with torch.no_grad():
            outputs = self.tsr_model(**encoding)

        target_sizes = [image.size[::-1]]
        results = self.feature_extractor.post_process_object_detection(outputs, threshold=0.6, target_sizes=target_sizes)[0]
        return results
    
    def call_result_to_voc_objects(self, results: dict, shift_coords: tuple[int, int] = [0, 0]) -> list[VocObject]:
        detected_objects = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            xmin, ymin, xmax, ymax = box
            xmin += shift_coords[0]
            ymin += shift_coords[1]
            xmax += shift_coords[0]
            ymax += shift_coords[1]
            detected_object = VocObject(category=self.id2label[label.item()],
                                        xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
                                        confidence=score.item())
            detected_objects.append(detected_object)

        return detected_objects

    def call_result_to_voc_layout(self, image, results: dict, image_name: str) -> VocLayout:
        detected_objects = self.call_result_to_voc_objects(results)

        height, width = image.size

        voc_layout = VocLayout(objects=detected_objects,
                               width=width, height=height, depth=3,
                               table_id=os.path.splitext(os.path.basename(image_name))[0])
        return voc_layout

    def plot_categories(self, image, results, categories: list[str]) -> np.ndarray:
        if isinstance(image, Image.Image):
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            label = self.id2label[label.item()]
            if categories and label not in categories:
                continue

            box = [round(i, 2) for i in box.tolist()]
            xmin, ymin, xmax, ymax = box
            color = (0, 0, 255)
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
            cv2.putText(image, f"{label}: {score:.2f}", (int(xmin), int(ymin) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return image

def parse_args():
    args = argparse.ArgumentParser()
    # args.add_argument("--model", type=str, default="microsoft/table-transformer-detection")
    # args.add_argument("--device", type=str, default="cuda")
    args.add_argument("-i", "--image-folder", type=str, default="example_data/tables")
    args.add_argument("-r", "--rendered", type=str, default="example_data/tables_tsr_renders",
                      help="Folder to save rendered images")

    return args.parse_args()


def main():
    args = parse_args()

    if args.rendered is not None:
        os.makedirs(args.rendered, exist_ok=True)


    # list all files in the image folder
    image_files = os.listdir(args.image_folder)
    image_extensions = ['.png', '.jpg', '.jpeg']
    image_files = [f for f in image_files if os.path.splitext(f)[1].lower() in image_extensions]

    tsr_engine = TableStructureRecognitionEngine()

    for image_file in image_files:
        print(f'Processing {image_file}...')
        image_path = os.path.join(args.image_folder, image_file)
        image = Image.open(image_path).convert("RGB")

        results = tsr_engine(image)
        # print(f'tsr result: {results}')

        voc_layout = tsr_engine.call_result_to_voc_layout(image, results, image_file)
        # print(f'voc layout: {voc_layout}')
        table_id = voc_layout.table_id

        image_columns = tsr_engine.plot_categories(np.array(image), results, ["table column"])
        cv2.imwrite(os.path.join(args.rendered, f"{table_id}_columns.png"), image_columns)
        print(f'\t-image with columns: {os.path.join(args.rendered, f"{table_id}_columns.png")}')
        image_rows = tsr_engine.plot_categories(np.array(image), results, ["table row", "table projected row header"])
        cv2.imwrite(os.path.join(args.rendered, f"{table_id}_rows.png"), image_rows)
        print(f'\t-image with rows: {os.path.join(args.rendered, f"{table_id}_rows.png")}')
        image_other = tsr_engine.plot_categories(np.array(image), results, ["table spanning cell", "table column header", "table"])
        cv2.imwrite(os.path.join(args.rendered, f"{table_id}_other.png"), image_other)
        print(f'\t-image with other: {os.path.join(args.rendered, f"{table_id}_other.png")}')


if __name__ == "__main__":
    main()
