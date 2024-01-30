from ultralytics import YOLO
import cv2 as cv

#paths
tracker_path = 'data/botsort.yaml'
model_path = 'models/pill.pt'

#initialize model
model = YOLO(model_path)

#Visualize results
results = model(source=0, show=True, conf=0.3, tracker=tracker_path)

cap = cv.VideoCapture(0)

while True:
    ret, frame = cap.read()
    results = model.track(frame, tracker=tracker_path)
    print(results)

    for result in results:
        print(result)
        
   
    cv.imshow('Pill Identification', frame)

    if cv.waitKey(1) > 0:
        break 