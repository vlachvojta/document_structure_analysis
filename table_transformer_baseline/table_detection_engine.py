import os
import sys

import cv2
import numpy as np

from huggingface_hub import hf_hub_download
from PIL import Image
# from transformers import DetrFeatureExtractor
from transformers import DetrImageProcessor
from transformers import TableTransformerForObjectDetection
import torch

from pubtables_1m.voc_xml import VocObject


class TableDetectionEngine:
    """Use microsoft table transformer model to detect tables in images."""

    def __init__(self, model_name: str = "microsoft/table-transformer-detection", device: str = "cuda"):
        # self.model_name = model_name
        # self.device = device
        # self.model = self.load_model()
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

        xmin, ymin, xmax, ymax = results["boxes"][0]
        xmin_ref = 203.07
        ymin_ref = 210.85
        xmax_ref = 1120.87
        ymax_ref = 384.19
        print(f'xmin:     {xmin:.2f} ymin:     {ymin:.2f}, xmax:     {xmax:.2f}, ymax:     {ymax:.2f}')
        print(f'xmin_ref: {xmin_ref:.2f} ymin_ref: {ymin_ref:.2f}, xmax_ref: {xmax_ref:.2f}, ymax_ref: {ymax_ref:.2f}')

        return results



    def render_results(self, image, results):
        for box, label, score in zip(results["boxes"], results["labels"], results["scores"]):
            # Convert the box coordinates to integers
            box = [round(i, 2) for i in box.tolist()]
            color = (255, 0, 0)
            # blue
            cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
            cv2.putText(image, f"{self.detection_model.config.id2label[label.item()]}: {round(score.item(), 3)}", (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # Visualize the detected tables on the image
        # render detections to the image
        return image

def main():
    input_image_path = os.path.join('example_data', 'pages', 'printed_page_1.png')
    image = Image.open(input_image_path).convert("RGB")
    width, height = image.size

    image.resize((int(width*0.5), int(height*0.5)))

    # feature_extractor = DetrImageProcessor()
    # model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-transformer-detection")

    # encoding = feature_extractor(image, return_tensors="pt")
    # print(encoding['pixel_values'].shape)
    # with torch.no_grad():
    #     outputs = model(**encoding)
    # results = feature_extractor.post_process_object_detection(outputs, threshold=0.7, target_sizes=[(height, width)])[0]

    # print(f'results: {results}')


    table_detection_engine = TableDetectionEngine()
    results = table_detection_engine(image)

    # pil image to cv2 image
    image_np = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    rendered_image = table_detection_engine.render_results(image_np, results)
    # save image to output.png
    output_path = 'output.png'
    cv2.imwrite(output_path, rendered_image)

    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

    # draw boxes on image
    for score, label, box in zip(results["scores"], results["labels"], results["boxes"]):
        box = [round(i, 2) for i in box.tolist()]
        color = (255, 0, 0)  # blue
        cv2.rectangle(image, (int(box[0]), int(box[1])), (int(box[2]), int(box[3])), color, 2)
        cv2.putText(image, f"{table_detection_engine.id2label[label.item()]}: {round(score.item(), 3)}", (int(box[0]), int(box[1]) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        # cv2 save image to output.png
        output_path = os.path.join('example_data', 'detection_output', 'output.png')
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        cv2.imwrite(output_path, image)

        # generate VocObject
        voc_obj = VocObject(category=table_detection_engine.id2label[label.item()],
            xmin=box[0], ymin=box[1], xmax=box[2], ymax=box[3],
            confidence=score.item())

        print(f'voc_obj: {voc_obj}') 

if __name__ == "__main__":
    main()
