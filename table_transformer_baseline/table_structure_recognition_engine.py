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

    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = self.load_model()

    def __call__(self, image) -> list[list[float]]:
        # see guide in the second part of
        # Using_Table_Transformer_for_table_detection_and_table_structure_recognition.ipynb

        # Perform table structure recognition on the input image
        # should return whatever is the output of the model
        # scores, labels, bboxes = self.model(image)
        # return scores, labels, bboxes
        pass

    def load_model(self):
        # Load the model from the specified path or URL
        pass

    def visualize(self, image, detections):
        # Visualize the detected tables on the image
        # render detections to the image
        pass

    def recognize_voc_layout(self, image) -> VocLayout:
        # Recognize the table structure in the image and return a VOC layout

        # results = self(image)  # get table structure recognition results
        # Process the results to create a VOC layout
        # return voc_layout
        pass

    # def recognize_page_layout(self, image) -> PageLayout:
    #     # Recognize the page layout in the image and return a page layout

    #     # results = self(image)  # get table structure recognition results
    #     # Process the results to create a page layout
    #     # return page_layout
    #     pass

# Example usage:
# engine = TableDetectionEngine(model_name="microsoft/table-transformer")

# TODO add example_data WITH expected output, and script main to run it..


def main():

    file_path = hf_hub_download(repo_id="nielsr/example-pdf", repo_type="dataset", filename="example_table.png")
    image = Image.open(file_path).convert("RGB")

    width, height = image.size
    image.resize((int(width*0.5), int(height*0.5)))
    # save image
    image.save("example_data/tables/example_table.png")


    feature_extractor = DetrImageProcessor()

    encoding = feature_extractor(image, return_tensors="pt")

    model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-structure-recognition")

    with torch.no_grad():
        outputs = model(**encoding)

    target_sizes = [image.size[::-1]]
    results = feature_extractor.post_process_object_detection(outputs, threshold=0.6, target_sizes=target_sizes)[0]

    detected_objects = []
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        xmin, ymin, xmax, ymax = box
        detected_object = VocObject(category=model.config.id2label[label.item()],
                                    xmin=xmin, ymin=ymin, xmax=xmax, ymax=ymax,
                                    confidence=score.item())
        detected_objects.append(detected_object)

    print(f'detected objects:')
    for detected_object in detected_objects:
        print(f'\t{detected_object}')

if __name__ == "__main__":
    main()
