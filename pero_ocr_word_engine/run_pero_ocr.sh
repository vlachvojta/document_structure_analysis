#!/bin/bash

# Author: Vojtěch Vlach
# Brief: Run PERO-OCR inference on specified folder of images
# Date: 10.10.2024
# Arguments:
#   $1: path to folder of images to infer
#   $2: path to config file (default: pipeline_layout_sort_ocr_czech_handwritten.ini)
# Usage: ./run_pero_ocr.sh /path/to/images /path/to/config.ini
# Description: Saves resulting data (page xmls and renders) in folders inside the images folder (for example /path/to/images -> /path/to/images/xml, /path/to/images/render)

# return $2 if $1 is empty (return default value if argument is not provided)
get_arg_or_default() {
    if ! [ -z "$1" ]; then
        echo $1
    else
        echo $2
    fi
}

SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)

IMAGES=$(get_arg_or_default $1 ./)
if ! [ -d "$IMAGES" ]; then
    echo "Directory $IMAGES does not exist. Provide a valid path as the first argument."
    exit 1
fi

CONFIG=$(get_arg_or_default $2 $SCRIPT_DIR/pipeline_layout_sort_ocr_czech_handwritten.ini)
if ! [ -f "$CONFIG" ]; then
    echo "Config file $CONFIG does not exist. Provide a valid path as the second argument."
    exit 1
fi

RENDER_PATH=${IMAGES}/render
RENDER_LINE_PATH=${IMAGES}/render_line
XML_PATH=${IMAGES}/xml

source ~/.env_pero/bin/activate

files_in_image_dir=$(ls -p -1 $IMAGES | grep -v / | wc -l)
echo "Reading ${files_in_image_dir} image files from ${IMAGES}":
echo ""

python $PERO_OCR/user_scripts/parse_folder.py \
    -c $CONFIG \
    -i $IMAGES \
    --output-xml-path $XML_PATH \
    --output-render-path $RENDER_PATH \
    --output-render-category \

# print info about input and output files
files_exported=$(ls -p -1 $XML_PATH/*.xml | wc -l)

if [ -d "$XML_PATH" ]; then
    echo -e "${GREEN}Exported ${files_exported} files from ${files_in_image_dir} images${ENDCOLOR}"
else
    echo -e "${RED}No XML files were exported${ENDCOLOR}"
fi

