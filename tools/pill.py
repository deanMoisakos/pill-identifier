from ultralytics import YOLO
import cv2 as cv
import numpy as np
import torch

class PillIdentifier:
    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print(f'Using Device: {self.device}')

        self.model = self.load_model()

    def load_model(self):
        model_path = 'models/pill.pt'
        model = YOLO(model_path)
        model.fuse()

        return model
    
    def predict(self, frame):

        results = self.model(frame)

        return results
    
    def plot_bboxes(self, results, frame):

        xyxys = []
        names = []
        confidences = []
        class_ids = []

        for result in results:
            boxes = result.boxes.cpu().numpy()

            if result.boxes.cls is None:
                    print('No Detections')
                    if cv.waitKey(1) > 0:
                        break
                    continue
            
            xyxys = boxes.xyxy
            names = result.names
            confidences = result.boxes.conf
            class_ids = result.boxes.cls
            
            for name, xyxy, conf_tensor, id_tensor in zip(names, xyxys, confidences, class_ids):

                #get id
                id = int(id_tensor.item())
                id_text = str(id)
                cv.putText(frame, id_text, (int(xyxy[0]), int(xyxy[1] - 100)), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)

                #display names
                name_text = str(result.names[name])  
                cv.putText(frame, name_text, (int(xyxy[0]), int(xyxy[1] - 5)), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)   

                #display confidence
                conf_float = (float(conf_tensor.item())*100)
                conf = "{:.0f}".format(conf_float) + "%"
                conf_text = str(conf)
                cv.putText(frame, conf_text, (int(xyxy[0]), int(xyxy[1] - 50)), cv.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 0), 4)
                
                #draw rectangle
                cv.rectangle(frame, (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])), (0, 255, 0), 2)
        
        return frame
    
    def __call__(self):
        cap = cv.VideoCapture(self.capture_index)
        assert cap.isOpened()

        while True:
            ret, frame = cap.read()
            if not ret:
                print("No Frame")
                break

            results = self.predict(frame)
            frame = self.plot_bboxes(results, frame)
            cv.imshow('frame', frame)

            if cv.waitKey(1) > 0:
                break