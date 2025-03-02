# Description: This script reads a Label Studio JSON file and prints the image sources of tasks that do not have an 'OK' choice selected or have text in the textarea (HTML annotation).

import json
import os
import sys
from typing import Any, Dict, List, Tuple

# add parent directory to python file path to enable imports
file_dirname = os.path.dirname(os.path.abspath(__file__))
sys.path.append(file_dirname)
sys.path.append(os.path.dirname(file_dirname))

from dataset.label_studio_utils import LabelStudioResults, label_studio_coords_to_xywh, add_padding

label_file = '/home/xvlach22/projects/DP/dataset_custom/1st_phase_detection_5_fixes/project-49-at-2025-02-17-14-38-8f8cebb4.json'

tasks = LabelStudioResults(label_file)

print(f'read {len(tasks)} tasks from {label_file}')
printed_images = 0

for task in tasks:
    if not len(task['annotations'])  == 1:
        print(f"task {task['id']} has {len(task['annotations'])} annotations, instead of 1, skipping")
        continue

    results = task['annotations'][0]['result']
    is_ok_result = False
    has_text = False

    for result in results:
        if result['type'] == 'choices':
            if result['value']['choices'] == ['OK']:
                is_ok_result = True
                continue
        if result['type'] == 'textarea':
            if result['value']['text']:
                has_text = True
                continue

    if not is_ok_result or has_text:
        # print image source
        print(task['data']['image'])
        printed_images += 1

print(f"printed {printed_images} images")