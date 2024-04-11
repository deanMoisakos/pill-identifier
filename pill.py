from ultralytics import YOLO
import cv2
import math

def video_detection(path_x):
    video_capture = path_x

    #create webcam
    cap=cv2.VideoCapture(video_capture)
    frame_width=int(cap.get(3))
    frame_height=int(cap.get(4))

    #initialize the model and classes
    model=YOLO("models/pill.pt")
    classNames=['Advil', 'Gravol', 'Tylenol 500', 'Tylenol Cold and Flu Nighttime', 'Tylenol Extra Strength']

    #main loop
    while True:
        success, img = cap.read()
        if not success:
            yield [None, []]
        results=model(img,stream=True)
        classStrings = ""
        for r in results:
            boxes=r.boxes
            for box in boxes:
                x1,y1,x2,y2=box.xyxy[0]
                x1,y1,x2,y2=int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(img, (x1,y1), (x2,y2), (255,0,255),3)
                conf=math.ceil((box.conf[0]*100))/100
                cls=int(box.cls[0])
                class_name=classNames[cls]
                label=f'{class_name}{conf}'
                print_label = f'Pill: {class_name} confidence: {conf}'
                if class_name == 'Advil':
                    print_label = f' Advil Detected | Purpose: Pain Relief | Dosage: 400mg Ibuprofen'
                if class_name == 'Tylenol 500':
                    print_label = f' Tylenol 500 Detected | Purpose: Pain Relief | Dosage: 500mg Acetaminophen'
                if class_name == 'Tylenol Extra Strength':
                    print_label = f' Tylenol Extra Strength Detected | Purpose: Pain Relief | Dosage: 500mg Acetaminophen'
                t_size = cv2.getTextSize(label, 0, fontScale=1, thickness=2)[0]
                c2 = x1 + t_size[0], y1 - t_size[1] - 3
                cv2.rectangle(img, (x1,y1), c2, [255,0,255], -1, cv2.LINE_AA)  # filled
                cv2.putText(img, label, (x1,y1-2),0, 1,[255,255,255], thickness=1,lineType=cv2.LINE_AA)
                classStrings = classStrings + "\n" + print_label 
        yield [img, classStrings]
cv2.destroyAllWindows()