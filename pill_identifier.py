from ultralytics import YOLO
import cv2 as cv
import numpy as np

#paths
tracker_path = 'models/botsort.yaml'
model_path = 'models/pill.pt'
pill_scatter = 'images/test_images/pill_scatter.PNG'
pill_contain = 'images/test_images/pill_contain.PNG'
pill_assorted = 'images/test_images/pill_assorted.PNG'
pill_napkin = 'images/test_images/pill_napkin.PNG'
pill_shadow = 'images/test_images/pill_shadow.PNG'
pill_video_1 = 'videos/pill_video_1.mp4'
pill_video_2 = 'videos/pill_video_2.mp4'

#input source
cap = cv.VideoCapture(pill_video_2)
#cap = cv.imread(pill_assorted)

#initialize model
model = YOLO(model_path)

#Visualize results
#results = model(source=pill_video_2, show=True, conf=0.3, tracker=tracker_path, imgsz = 640)

while True:

    ret, frame = cap.read()
    if not ret:
        print("No Frame")
        break

    

    results = model.track(frame, conf=0.5, tracker=tracker_path, imgsz = 640 ,verbose=False)

    for r in results:

        if r.boxes.id is None:
                    print('No Detections')
                    if cv.waitKey(1) > 0:
                        break
                    continue 

        for id_tensor, box, name, conf_tensor in zip(r.boxes.id, r.boxes.xyxy, r.names, r.boxes.conf):
            
            #get id
            id = int(id_tensor.item())

            print(id)

            #draw bounding box
            x1, y1, x2, y2 = box[:4]
            cv.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            #display id
            text = str(id)
            cv.putText(frame, text, (int(x1), int(y1 + 40)), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)

            #display names
            text = str(r.names[name])       
            cv.putText(frame, text, (int(x1), int(y1 - 2)), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)

            #display confidence
            conf_float = (float(conf_tensor.item())*100)
            conf = "{:.0f}".format(conf_float) + "%"
            text = str(conf)
            cv.putText(frame, text, (int(x2 - 50), int(y1 - 2)), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)

    cv.imshow('Pill Identification', frame)

    if cv.waitKey(1) > 0:
            break

cap.release()