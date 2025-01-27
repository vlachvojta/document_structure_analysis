# Description: This script converts the annotations json file to YOLO txt files.
# This code is based on the homework of the "Computer Vision" course (POVa) at FIT VUT. (Brno University of Technology)
# Authors: Martin Kostelník and Michal Hradiš (2024)
# Contributors: Vojtěch Vlach (2024-2025)
# coding: utf-8

import os
import argparse
import json
import yaml


def parse_arguments():
    parser = argparse.ArgumentParser(description="Inference script for video object detection using pre-trained YOLO model.")

    parser.add_argument("-i", "--images", type=str, required=True, help="Path to the images directory.")
    parser.add_argument("-a", "--annotations", type=str, required=True, help="Path to the annotation json file.")
    parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output directory.")
    parser.add_argument("-d", "--data-yaml", type=str, required=True, help="Path to the data yaml file.")

    return parser.parse_args()


def main():
    args = parse_arguments()

    if not os.path.isdir(args.images):
        raise FileNotFoundError(f"Error: Unable to find images directory at {args.images}.")
    if not os.path.isfile(args.annotations):
        raise FileNotFoundError(f"Error: Unable to find annotations json file at {args.annotations}.")
    if not os.path.isfile(args.data_yaml):
        raise FileNotFoundError(f"Error: Unable to find data yaml file at {args.data_yaml}.")

    os.makedirs(args.output, exist_ok=True)

    stats = {
        "missing_images": 0,
        "unknown_labels": 0,
        "total_annotations": 0,
        "total_objects": 0,
        "images_ok": 0,
        "objects_ok": 0,
    }

    # load data yaml
    with open(args.data_yaml, "r") as file:
        data = yaml.safe_load(file)

    id_to_label = data["names"]
    label_to_id = {label: id for id, label in id_to_label.items()}
    print(f"Loaded {len(label_to_id)} labels. {label_to_id}")

    # load images
    images = os.listdir(args.images)
    images = set(images)

    # load annotations
    with open(args.annotations, "r") as file:
        annotations = json.load(file)

    # create YOLO txt files
    for annotation in annotations:
        stats["total_annotations"] += 1
        image_name = annotation["image"]
        # check if image exists
        if image_name not in images:
            stats["missing_images"] += 1
            continue

        image_name = os.path.splitext(image_name)[0]

        objects = annotation["label"]

        with open(os.path.join(args.output, f"{image_name}.txt"), "w") as file:
            for obj in objects:
                stats["total_objects"] += 1
                try:
                    label = label_to_id[obj["rectanglelabels"][0]]
                except KeyError:
                    stats["unknown_labels"] += 1
                    print(f'WARNING: Unknown label "{obj["rectanglelabels"][0]}" in image {image_name}. Skipping...')
                    continue
                x = obj["x"] + (obj["width"] / 2)
                x /= 100
                y = obj["y"] + (obj["height"] / 2)
                y /= 100

                width = obj["width"] / 100
                height = obj["height"] / 100

                file.write(f"{label} {x} {y} {width} {height}\n")
                stats["objects_ok"] += 1
        stats["images_ok"] += 1

    print(f"Finished creating YOLO txt files. ")
    print(f"Stats:")
    print(json.dumps(stats, indent=4))

if __name__ == "__main__":
    main()

# example of wanted YOLO txt file
# 1 0.123 0.456 0.789 0.101
# 0 0.123 0.456 0.789 0.101


# example of annotations json file
# [
#   {
#     "image": "20241106_133732.jpg",
#     "id": 142825,
#     "label": [
#       {
#         "x": 29.28719008264463,
#         "y": 31.404958677685947,
#         "width": 5.88842975206612,
#         "height": 11.157024793388432,
#         "rotation": 0,
#         "rectanglelabels": [
#           "mouse"
#         ],
#         "original_width": 1024,
#         "original_height": 576
#      },
#       {
#         "x": 26.34297520661157,
#         "y": 0,
#         "width": 19.44731404958678,
#         "height": 22.31404958677686,
#         "rotation": 0,
#         "rectanglelabels": [
#           "keyboard"
#         ],
#         "original_width": 1024,
#         "original_height": 576
#       }
#     ],
#     "annotator": 20,
#     "annotation_id": 22415,
#     "created_at": "2024-11-06T12:59:48.587825Z",
#     "updated_at": "2024-11-06T13:08:18.032650Z",
#     "lead_time": 48.026
#   },
#   {
#     "image": "20241106_133735.jpg",
#     "id": 142826,
#     "label": [
#       {
#         "x": 0,
#         "y": 26.836592690251187,
#         "width": 38.23335530652604,
#         "height": 17.344173441734455,
#         "rotation": 0,
#         "rectanglelabels": [
#           "keyboard"
#         ],
#         "original_width": 1024,
#         "original_height": 576
#       },
#       {
#         "x": 1.5161502966381015,
#         "y": 47.22771552039845,
#         "width": 7.514831905075808,
#         "height": 14.297224053321614,
#         "rotation": 0,
#         "rectanglelabels": [
#           "mouse"
#         ],
#         "original_width": 1024,
#         "original_height": 576
#       }
#     ]
#   }
# ]