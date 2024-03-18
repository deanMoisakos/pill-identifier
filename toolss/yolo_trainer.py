from ultralytics import YOLO

yolo = YOLO()

yolo.train(model='models/yolov8m.pt', data='data/data.yaml', epochs=200)