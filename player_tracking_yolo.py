import cv2
import numpy as np

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
# Print layer_names

unconnected_out_layers = net.getUnconnectedOutLayers()

print("Layer names:")
print(layer_names)

# Print unconnected_out_layers
print("Unconnected out layers:")
print(unconnected_out_layers)

# output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]
output_layers = []
for i in unconnected_out_layers:
    print("i")
    print(i)

    index = i - 1  # Adjusting for 0-based indexing in Python
    layer_name = layer_names[index]
    output_layers.append(layer_name)

classes = []
with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Load video
cap = cv2.VideoCapture('soccer_training.mp4')

# # Initialize tracker with first frame and bounding box
# ret, frame = cap.read()
# # bbox = (287, 23, 86, 320)  # You need to select the player using a bounding box.
# bbox = (800, 300, 400, 320)  # You need to select the player using a bounding box.

# Read first frame
ret, frame = cap.read()

# Select bounding box
bbox = cv2.selectROI(frame, False)
cv2.destroyAllWindows()

# tracker = cv2.TrackerCSRT_create()
tracker = cv2.TrackerMIL_create()

tracker.init(frame, bbox)

while True:
    ret, img = cap.read()
    
    height, width, channels = img.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
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
                tracker.update(img)

                # Draw bounding box
                p1 = (int(bbox[0]), int(bbox[1]))
                p2 = (int(bbox[0] + bbox[2]), int(bbox[1] + bbox[3]))
                cv2.rectangle(img, p1, p2, (255,0,0), 2, 1)

    cv2.imshow("Image", img)

    # Exit if ESC key is pressed
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the VideoCapture and destroy all windows
cap.release()
cv2.destroyAllWindows()
