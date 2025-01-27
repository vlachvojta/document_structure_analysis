# Description: This script converts the annotations json file to YOLO txt files.
# This code is based on the homework of the "Computer Vision" course (POVa) at FIT VUT. (Brno University of Technology)
# Authors: Martin Kostelník and Michal Hradiš (2024)
# Contributors: Vojtěch Vlach (2024-2025)
# coding: utf-8

import os
import argparse
import json
import yaml
from tqdm import tqdm


def parse_arguments():
    parser = argparse.ArgumentParser(description="Inference script for video object detection using pre-trained YOLO model.")

    parser.add_argument("-i", "--images", type=str, default="example_data/images",
                        help="Path to the images directory.")
    parser.add_argument("-a", "--annotations", type=str, default="example_data/annotations_label_studio.json",
                        help="Path to the annotation json file.")
    parser.add_argument("-o", "--output", type=str, default="example_data/labels",
                        help="Path to the output directory.")
    parser.add_argument("-d", "--data-yaml", type=str, default="example_data/data.yaml",
                        help="Path to the data yaml file.")

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
        "unknown_labels": {}, # label: count
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
    # for annotation in :
    for annotation in tqdm(annotations):
        stats["total_annotations"] += 1
        image_address = annotation["data"]["image"]
        image_name = image_address.split('/')[-1]
        # check if image exists
        if image_name not in images:
            stats["missing_images"] += 1
            continue

        image_name = os.path.splitext(image_name)[0]
        objects = annotation["annotations"][0]["result"]

        with open(os.path.join(args.output, f"{image_name}.txt"), "w") as file:
            for obj in objects:
                stats["total_objects"] += 1
                label = obj["value"]["rectanglelabels"][0]
                if label not in label_to_id:
                    stats["unknown_labels"].setdefault(label, 0)
                    stats["unknown_labels"][label] += 1
                    print(f'Unknown label "{label}" in image {image_name}. Skipping...')
                    continue
                label_id = label_to_id[label]

                x = obj["value"]["x"] + (obj["value"]["width"] / 2)
                x /= 100
                y = obj["value"]["y"] + (obj["value"]["height"] / 2)
                y /= 100

                width = obj["value"]["width"] / 100
                height = obj["value"]["height"] / 100

                file.write(f"{label_id} {x} {y} {width} {height}\n")
                stats["objects_ok"] += 1
        stats["images_ok"] += 1

    print(f"Finished creating YOLO txt files. ")
    print(f"Stats:")
    print(json.dumps(stats, indent=2))

if __name__ == "__main__":
    main()

# example of wanted YOLO txt file
# 1 0.123 0.456 0.789 0.101
# 0 0.123 0.456 0.789 0.101
