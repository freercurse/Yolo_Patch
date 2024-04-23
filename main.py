from ultralytics import YOLO
import cv2 as cv
from captum.attr import Occlusion

#initialising the YOLO model and webcam capture references
model = YOLO("model/yolov8n.pt")

cap = cv.VideoCapture(0)

if not cap.isOpened():
 print("Cannot open camera")
 exit()


def placeBoxes(frame, preds):
    """Takes the output from the YOLO prediction generators and places bboxes and labels against xyxy pred coords"""
    final = frame
    
    for results in preds:        
        if(len(results) > 0):
            for i, objs in enumerate(results.boxes.xyxy.numpy()):                
                objClass = int(results.boxes.cls[i].numpy())
                
                final = cv.rectangle(frame, (int(objs[0]), int(objs[1])), (int(objs[2]), int(objs[3])), (0,255,0), 3)
                cv.putText(final,results.names.get(objClass), (int(objs[0]), int(objs[1]- 10)), cv.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        
        print(len(results))

    return final




# main frame prediction loop
while(1):

    ret, frame = cap.read()

    if cv.waitKey(1) == ord('q'):
        break

    results = model.predict(source=frame, stream=True)
    preds = placeBoxes(frame, results)

    cv.imshow("webcam", preds)

#cleanup
cap.release()
cv.destroyAllWindows()