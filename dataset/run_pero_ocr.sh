#!/bin/bash

# Author: Vojtěch Vlach
# Brief: Run PERO-OCR inference on specified folder of images
# Date: 10.10.2024
# Arguments:
#   $1: path to folder of images to infer
# Usage: ./run_pero_ocr.sh /path/to/images
# Description: Saves resulting data (page xmls and renders) in folders next to the images (for example /path/to/images -> /path/to/images-xml, /path/to/images-render)

# return $2 if $1 is empty (return default value if argument is not provided)
get_arg_or_default() {
    if ! [ -z "$1" ]; then
        echo $1
    else
        echo $2
    fi
}

# get abspath of this script
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

IMAGES=$(get_arg_or_default $1 ./)
if ! [ -d "$IMAGES" ]; then
    echo "Directory $IMAGES does not exist. Provide a valid path as the first argument."
    exit 1
fi

RENDER_PATH=${IMAGES}/render
RENDER_LINE_PATH=${IMAGES}/render_line
XML_PATH=${IMAGES}/xml

source ~/.env_pero/bin/activate

# Generate xmls
# python $PERO_OCR/user_scripts/parse_folder.py -c pipeline_config_double_layout_parser_and_engines_cpu.ini -i images --output-render-path output_render --output-line-path output_line/ --output-xml-path xmls --output-alto-path xmls --output-logit-path logits --device cpu  # CPU

files_in_image_dir=$(ls -p -1 $IMAGES | grep -v / | wc -l)
echo "Reading ${files_in_image_dir} image files from ${IMAGES}":
echo ""

python $PERO_OCR/user_scripts/parse_folder.py \
    -c $SCRIPT_DIR/pipeline_layout_sort_ocr.ini \
    -i $IMAGES \
    --output-xml-path $XML_PATH \
    --output-line-path $RENDER_LINE_PATH \
    --output-render-path $RENDER_PATH \
    --output-render-category \

# print info about input and output files
files_exported=$(ls -p -1 $XML_PATH/*.xml | wc -l)

if [ -d "$XML_PATH" ]; then
    echo -e "${GREEN}Exported ${files_exported} files from ${files_in_image_dir} images${ENDCOLOR}"
else
    echo -e "${RED}No XML files were exported${ENDCOLOR}"
fi

