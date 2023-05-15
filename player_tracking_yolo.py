import cv2
import numpy as np

def load_yolo():
    net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
    layer_names = net.getLayerNames()
    unconnected_out_layers = net.getUnconnectedOutLayers()
    output_layers = []
    for i in unconnected_out_layers:
        print("i")
        print(i)

        index = i - 1  # Adjusting for 0-based indexing in Python
        layer_name = layer_names[index]
        output_layers.append(layer_name)
    return net, output_layers

def select_bounding_box(frame):
    bbox = cv2.selectROI(frame, False)
    cv2.destroyAllWindows()
    return bbox

def initialize_tracker(frame, bbox):
    tracker = cv2.TrackerCSRT_create()
    ok = tracker.init(frame, bbox)
    return tracker, ok

def track_object(tracker, frame):
    ok, bbox = tracker.update(frame)
    if ok:
        p1 = (int(bbox[0]), int(bbox[1]))
        p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
        cv2.rectangle(frame, p1, p2, (255, 0, 0), 2, 1)
    cv2.imshow("Tracking", frame)
    return ok


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    net, output_layers = load_yolo()

    classes = []
    with open("coco.names", "r") as f:
        classes = [line.strip() for line in f.readlines()]

    while True:
     ret, frame = cap.read()
     if not ret:
            break
     
     height, width, channels = frame.shape

     # Detecting objects
     blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
     net.setInput(blob)
     outs = net.forward(output_layers)

     # Showing informations on the screen
     class_ids = []
     confidences = []
     boxes = []
     for out in outs:
         for detection in out:
             scores = detection[5:]
             class_id = np.argmax(scores)
             confidence = scores[class_id]
             if confidence > 0.2:
                 # Object detected
                 center_x = int(detection[0] * width)
                 center_y = int(detection[1] * height)
                 w = int(detection[2] * width)
                 h = int(detection[3] * height)

                 # Rectangle coordinates
                 x = int(center_x - w / 2)
                 y = int(center_y - h / 2)

                 boxes.append([x, y, w, h])
                 confidences.append(float(confidence))
                 class_ids.append(class_id)

     indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
     for i in range(len(boxes)):
         if i in indexes:
             x, y, w, h = boxes[i]
             label = str(classes[class_ids[i]])
             if label == 'person':
                 # Update tracker
                 bbox = select_bounding_box(frame)
                 tracker, ok = initialize_tracker(frame, bbox)
                 tracker.update(frame)

                 # Draw bounding box
                 p1 = (int(bbox[0]), int(bbox[1]))
                 p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                 cv2.rectangle(frame, p1, p2, (255,0,0), 2, 1)

     cv2.imshow("Tracking", frame)

     # Exit if ESC key is pressed
     if cv2.waitKey(1) & 0xFF == 27:
         break

# Release the VideoCapture and destroy all windows
    cap.release()
    cv2.destroyAllWindows() 

process_video("soccer_training.mp4")