import os
import sys
import argparse

import cv2
import numpy as np

from huggingface_hub import hf_hub_download
from PIL import Image
from transformers import DetrImageProcessor, TableTransformerForObjectDetection
import torch

from pubtables_1m.voc_xml import VocObject


class TableDetectionEngine:
    """Use microsoft table transformer model to detect tables in images."""

    def __init__(self, model_name: str = "microsoft/table-transformer-detection", device: str = "cuda"):
        self.feature_extractor = DetrImageProcessor()
        self.detection_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")
        self.id2label = self.detection_model.config.id2label

    def __call__(self, image):
        width, height = image.size

        encoding = self.feature_extractor(image, return_tensors="pt")
        with torch.no_grad():
            outputs = self.detection_model(**encoding)
        results = self.feature_extractor.post_process_object_detection(
            outputs, threshold=0.7, target_sizes=[(height, width)])[0]

        return results

    def get_tables_as_voc_objects(self, image: Image.Image) -> list[VocObject]:
        results = self(image)
        tables = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]

            xmin, ymin, xmax, ymax = box
            table = VocObject(category=self.id2label[label.item()],
                xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
                confidence=score.item())
            tables.append(table)

        return tables

    def get_table_crops(self, image: Image.Image, results: dict, padding: int = 10) -> list[Image.Image]:
        crops = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            table_crop = image.crop((box[0] - padding,
                                     box[1] - padding,
                                     box[2] + padding,
                                     box[3] + padding))
            crops.append(table_crop)

        return crops

    def render_results(self, image, results) -> np.ndarray:
        # if image is pil, convert to cv2
        if isinstance(image, Image.Image):
            image = np.array(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
            # Convert the box coordinates to integers
            box = [int(round(i, 2)) for i in box.tolist()]
            color = (255, 0, 0) # blue
            cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), color, 2)
            cv2.putText(image, f"{self.detection_model.config.id2label[label.item()]}: {round(score.item(), 3)}", (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # Visualize the detected tables on the image
        # render detections to the image
        return image

def parse_args():
    args = argparse.ArgumentParser()
    # args.add_argument("--model", type=str, default="microsoft/table-transformer-detection")
    # args.add_argument("--device", type=str, default="cuda")
    args.add_argument("-i", "--image-folder", type=str, default="example_data/pages")
    args.add_argument("-t", "--table-crops", type=str, default="example_data/pages_crops")
    args.add_argument("-r", "--rendered", type=str, default="example_data/rendered")

    return args.parse_args()

def main():
    args = parse_args()

    if args.table_crops is not None:
        os.makedirs(args.table_crops, exist_ok=True)
    if args.rendered is not None:
        os.makedirs(args.rendered, exist_ok=True)

    # list all files in the image folder
    image_files = os.listdir(args.image_folder)
    image_extensions = ['.png', '.jpg', '.jpeg']
    image_files = [f for f in image_files if os.path.splitext(f)[1].lower() in image_extensions]

    table_detection_engine = TableDetectionEngine()

    for image_file in image_files:
        image_path = os.path.join(args.image_folder, image_file)
        image = Image.open(image_path).convert("RGB")
        width, height = image.size

        results = table_detection_engine(image)
        print(f'results: {results}')

        if len(results["scores"]) == 0:
            # no tables detected
            continue

        # save the image with boxes
        image_np = np.array(image)
        image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        rendered_results = table_detection_engine.render_results(image_np, results)
        render_out_file = os.path.join(args.rendered, image_file)
        cv2.imwrite(render_out_file, rendered_results)

        # save individual table crops
        crops = table_detection_engine.get_table_crops(image, results)
        for i, crop in enumerate(crops):
            crop_out_file = os.path.join(args.table_crops, f"{os.path.splitext(image_file)[0]}_crop_{i}.png")
            crop.save(crop_out_file)
            print(f"Saved crop to {crop_out_file}")


if __name__ == "__main__":

    main()
