# Description: This script is used to infer the YOLO model on the images in the specified directory.
# This code is based on the work of Martin Kišš in the Orbis Pictus project at FIT VUT (Brno University of Technology). The script was originally used for detection of non-text regions in historical documents.
# Original script is available as the first commit of this filein this repository.
# Author: Martin Kišš (2024)
# Contributor: Vojtěch Vlach (2025)
# coding: utf-8

import os
import cv2
import argparse

from collections import defaultdict
from ultralytics import YOLO

# from safe_gpu.safe_gpu import GPUOwner


# CROPPED_CATEGORIES = {"obrázek", "fotografie", "kreslený-humor-karikatura-komiks", "erb-cejch-logo-symbol", "iniciála", "mapa", "graf", "geometrické-výkresy", "ostatní-výkresy", "schéma", "půdorys", "ex-libris" }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model", default="table_det_yolov8n_640.pt",
                        help="Model path.") 
    parser.add_argument("-i", "--images", default="example_data/images",
                        help="Path to a directory with images.")
    parser.add_argument("-s", "--image-size", default=640, type=int,
                        help="Image size.")
    parser.add_argument("-b", "--batch-size", required=False, default=1, type=int)
    parser.add_argument("-t", "--confidence-threshold", default=0.5, type=float,
                        help="Detection confidence threshold.")
    parser.add_argument("-p", "--predictions", default='example_data/predictions',
                        help="Path to a directory with predicted predictions.")
    parser.add_argument("-c", "--crops", default='example_data/crops',
                        help="Path to a directory with cropped images.")
    parser.add_argument("-r", "--renders", default='example_data/renders',
                        help="Path to a directory with renders.")
    parser.add_argument("-d", "--device", choices=['cpu', 'gpu'], default= 'gpu',
                        help="Device to use")
    parser.add_argument("-e", "--export-empty", action='store_true', default=False,
                        help="Export empty labels and renders.")

    return parser.parse_args()


def load_image(path):
    return cv2.imread(path)


def save_image(path, image):
    cv2.imwrite(path, image)


def save_predictions(path, predictions):
    with open(path, "w") as file:
        for line in predictions:
            file.write(f"{line}\n")


def normalize_name(name):
    name = name.replace(" ", "-")
    name = name.replace("/", "-")
    name = name.lower()
    return name


def get_crop_output_path(original_image_path, crops_dir, label_name, label_index):
    _, filename = os.path.split(original_image_path)
    filename, _ = os.path.splitext(filename)
    crop_output_path = os.path.join(crops_dir, f"{filename}__{label_name}_{label_index}.jpg")
    return crop_output_path


def get_label_output_path(original_image_path, predictions_dir):
    _, filename = os.path.split(original_image_path)
    filename, _ = os.path.splitext(filename)
    predictions_path = os.path.join(predictions_dir, f"{filename}.txt")
    return predictions_path


def get_render_output_path(original_image_path, renders_dir):
    _, filename = os.path.split(original_image_path)
    render_output_path = os.path.join(renders_dir, filename)
    return render_output_path


def main():
    args = parse_args()

    if args.device == 'gpu':
        device = 0
    else:
        device = 'cpu'

    extensions = (".jpg", ".png")

    images = [f"{os.path.join(args.images, image)}" for image in os.listdir(args.images) if image.endswith(extensions)]
    model = YOLO(args.model)

    if args.predictions is not None and not os.path.exists(args.predictions):
        os.makedirs(args.predictions)

    if args.crops is not None and not os.path.exists(args.crops):
        os.makedirs(args.crops)

    if args.renders is not None and not os.path.exists(args.renders):
        os.makedirs(args.renders)

    while images:
        batch = images[:args.batch_size]
        images = images[args.batch_size:]

        results = model(batch, 
                        imgsz=args.image_size,
                        conf=args.confidence_threshold,
                        device=device)

        if args.predictions or args.crops or args.renders:
            for result in results:
                image = result.orig_img
                predictions = []
                predictions_counter = defaultdict(int)

                for label, bbox in zip(result.boxes.cls, result.boxes.xyxy):
                    name = result.names[label.item()]
                    name = normalize_name(name)
                    coords = [round(coord.item()) for coord in bbox]

                    # if args.crops and name in CROPPED_CATEGORIES:
                    if args.crops:
                        crop = image[coords[1]:coords[3], coords[0]:coords[2]]
                        crop_output_path = get_crop_output_path(original_image_path=result.path, crops_dir=args.crops, label_name=name, label_index=predictions_counter[name])
                        save_image(crop_output_path, crop)

                    predictions_counter[name] += 1

                    predictions.append(f"{name} {coords[0]} {coords[1]} {coords[2]} {coords[3]}")

                if args.predictions and (len(predictions) > 0 or args.export_empty):
                    label_output_path = get_label_output_path(original_image_path=result.path, predictions_dir=args.predictions)
                    save_predictions(label_output_path, predictions)

                if args.renders and (len(predictions) > 0 or args.export_empty):
                    render_output_path = get_render_output_path(original_image_path=result.path, renders_dir=args.renders)
                    render = result.plot()
                    save_image(render_output_path, render)

    return 0


if __name__ == "__main__":
    exit(main())
