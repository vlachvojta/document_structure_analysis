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

def plot_certain_categories(image: np.ndarray, results: dict, categories: list[str], id2label: dict) -> np.ndarray:
    # if image is a PIL image, convert to numpy array
    if isinstance(image, Image.Image):
        image = np.array(image)
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        label = id2label[label.item()]
        if label not in categories:
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

    output_render_folder = "example_data/tables_tsr_renders/"
    os.makedirs(output_render_folder, exist_ok=True)

    image_columns = plot_certain_categories(np.array(image), results, ["table column"], model.config.id2label)
    cv2.imwrite(os.path.join(output_render_folder, "columns.png"), image_columns)
    image_rows = plot_certain_categories(np.array(image), results, ["table row", "table projected row header"], model.config.id2label)
    cv2.imwrite(os.path.join(output_render_folder, "rows.png"), image_rows)
    image_other = plot_certain_categories(np.array(image), results, ["table spanning cell", "table column header", "table"], model.config.id2label)
    cv2.imwrite(os.path.join(output_render_folder, "other.png"), image_other)


if __name__ == "__main__":
    main()
