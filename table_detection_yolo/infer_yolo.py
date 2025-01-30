# Description: This script is used to infer the YOLO model on the images in the specified directory.
# This code is based on the work of Martin Kišš in the Orbis Pictus project at FIT VUT (Brno University of Technology). The script was originally used for detection of non-text regions in historical documents.
# Author: Martin Kišš (2024)
# coding: utf-8

import os
import cv2
import argparse

from collections import defaultdict
from ultralytics import YOLO

from safe_gpu.safe_gpu import GPUOwner


CROPPED_CATEGORIES = {"obrázek", "fotografie", "kreslený-humor-karikatura-komiks", "erb-cejch-logo-symbol", "iniciála", "mapa", "graf", "geometrické-výkresy", "ostatní-výkresy", "schéma", "půdorys", "ex-libris" }


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Model path.", required=True)
    parser.add_argument("--images", help="Path to a directory with images.", required=True)
    parser.add_argument("--image-size", help="Image size.", required=False, default=640, type=int)
    parser.add_argument("--batch-size", help="Batch size.", required=False, default=1, type=int)
    parser.add_argument("--confidence", help="Confidence threshold.", required=True, type=float)
    parser.add_argument("--labels", help="Path to a directory with predicted labels.", required=False)
    parser.add_argument("--crops", help="Path to a directory with cropped images.", required=False)
    parser.add_argument("--renders", help="Path to a directory with renders.", required=False)

    return parser.parse_args()


def load_image(path):
    return cv2.imread(path)


def save_image(path, image):
    cv2.imwrite(path, image)


def save_labels(path, labels):
    with open(path, "w") as file:
        for line in labels:
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


def get_label_output_path(original_image_path, labels_dir):
    _, filename = os.path.split(original_image_path)
    filename, _ = os.path.splitext(filename)
    labels_path = os.path.join(labels_dir, f"{filename}.txt")
    return labels_path


def get_render_output_path(original_image_path, renders_dir):
    _, filename = os.path.split(original_image_path)
    render_output_path = os.path.join(renders_dir, filename)
    return render_output_path


def main():
    args = parse_args()

    gpu_owner = GPUOwner()

    extensions = (".jpg", ".png")

    images = [f"{os.path.join(args.images, image)}" for image in os.listdir(args.images) if image.endswith(extensions)]
    model = YOLO(args.model)

    if args.labels is not None and not os.path.exists(args.labels):
        os.makedirs(args.labels)

    if args.crops is not None and not os.path.exists(args.crops):
        os.makedirs(args.crops)

    if args.renders is not None and not os.path.exists(args.renders):
        os.makedirs(args.renders)

    while images:
        batch = images[:args.batch_size]
        images = images[args.batch_size:]

        results = model(batch, 
                        imgsz=args.image_size,
                        conf=args.confidence,
                        device=0)
        

        if args.labels or args.crops or args.renders:
            for result in results:
                image = result.orig_img
                labels = []
                labels_counter = defaultdict(int)

                for label, bbox in zip(result.boxes.cls, result.boxes.xyxy):
                    name = result.names[label.item()]
                    name = normalize_name(name)
                    coords = [round(coord.item()) for coord in bbox]

                    if args.crops and name in CROPPED_CATEGORIES:
                        crop = image[coords[1]:coords[3], coords[0]:coords[2]]
                        crop_output_path = get_crop_output_path(original_image_path=result.path, crops_dir=args.crops, label_name=name, label_index=labels_counter[name])
                        save_image(crop_output_path, crop)
                    
                    labels_counter[name] += 1

                    labels.append(f"{name} {coords[0]} {coords[1]} {coords[2]} {coords[3]}")
                    
                if args.labels and len(labels) > 0:
                    label_output_path = get_label_output_path(original_image_path=result.path, labels_dir=args.labels)
                    save_labels(label_output_path, labels)

                if args.renders:
                    render_output_path = get_render_output_path(original_image_path=result.path, renders_dir=args.renders)
                    render = result.plot()
                    save_image(render_output_path, render)

    return 0


if __name__ == "__main__":
    exit(main())
