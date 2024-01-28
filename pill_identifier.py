from ultralytics import YOLO

model = YOLO('models/pill.pt')

results = model(source=0, show=True, conf=0.3, tracker='botsort.yaml')