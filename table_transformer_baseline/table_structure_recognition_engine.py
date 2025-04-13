import os
import sys
import argparse

import cv2
import numpy as np

from huggingface_hub import hf_hub_download
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
    
    def call_result_to_voc_objects(self, results: dict) -> list[VocObject]:
        detected_objects = []
        for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
            box = [round(i, 2) for i in box.tolist()]
            xmin, ymin, xmax, ymax = box
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
            cv2.rectangle(image, (int(xmin), int(ymin)), (int(xmax), int(ymax)), (0, 255, 0), 2)
            cv2.putText(image, f"{label}: {score:.2f}", (int(xmin), int(ymin) - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image


def main():
    file_path = hf_hub_download(repo_id="nielsr/example-pdf", repo_type="dataset", filename="example_table.png")
    image = Image.open(file_path).convert("RGB")

    width, height = image.size
    image.resize((int(width*0.5), int(height*0.5)))

    tsr_model = TableStructureRecognitionEngine()
    results = tsr_model(image)

    detected_objects = tsr_model.call_result_to_voc_objects(results)

    print(f'detected objects:')
    for detected_object in detected_objects:
        print(f'\t{detected_object}')

    output_render_folder = "example_data/tables_tsr_renders/"
    os.makedirs(output_render_folder, exist_ok=True)

    image_columns = tsr_model.plot_categories(np.array(image), results, ["table column"])
    cv2.imwrite(os.path.join(output_render_folder, "columns.png"), image_columns)
    image_rows = tsr_model.plot_categories(np.array(image), results, ["table row", "table projected row header"])
    cv2.imwrite(os.path.join(output_render_folder, "rows.png"), image_rows)
    image_other = tsr_model.plot_categories(np.array(image), results, ["table spanning cell", "table column header", "table"])
    cv2.imwrite(os.path.join(output_render_folder, "other.png"), image_other)

    voc_layout = tsr_model.call_result_to_voc_layout(image, results, "example_table.png")
    print(f'voc layout: {voc_layout}')


if __name__ == "__main__":
    main()
