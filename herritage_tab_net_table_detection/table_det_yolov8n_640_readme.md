# README for table_det_yolov8n_640.pt

Training command:
yolo detect train data=../data.yaml model=yolov8n.pt augment=True dropout=0.4 epochs=100 imgsz=640 project=xvlach_table_det_yolo name=yolov8n_augment batch=8

Trn data:
420 images of 5 categories (see data.yaml)

