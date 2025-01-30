#!/bin/bash

# This script is here to run scripts in the correct order. And give them the correct arguments.


# return $2 if $1 is empty (return default value if argument is not provided)
get_arg_or_default() {
    if ! [ -z "$1" ]; then
        echo $1
    else
        echo $2
    fi
}

WORKDIR=$(get_arg_or_default $1 "example_data")
DSA=$(get_arg_or_default $2 "/path/to/dataset_structure_analysis")


# 0 - input: images of pages + table annotations
# 1 - cutout table annotations
CUT_TABLE_DIR=$WORKDIR/1_cut_tables
python $DSA/dataset/cut_annotations.py -i $WORKDIR/0_images/ -l $WORKDIR/0_annotated_table_detection.json -o $CUT_TABLE_DIR -f "Table"

# OCR Page-XML of table annotations
$DSA/dataset/run_pero_ocr.sh $CUT_TABLE_DIR

# 2 - create tasks for tables with OCR TextLines as predictions 
python $DSA/dataset/ocr_to_tasks.py -i $CUT_TABLE_DIR -x $CUT_TABLE_DIR/xml -o $WORKDIR/2_cell_detection_tasks

