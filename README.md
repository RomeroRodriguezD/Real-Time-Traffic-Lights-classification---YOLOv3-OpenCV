# Real-Time-Traffic-Lights-classification---YOLOv3-OpenCV
Classifying traffic lights on a video streaming with OpenCV, deep neural network + classic computer vision using HSV channels to determine colors.
It uses an already downloaded YOLOv3 weights.


<h2> Main libraries: OpenCV and Threading. </h2>

```

import cv2
import numpy as np
from threading import Thread
from queue import Queue
import time

```

<h2> Setting up YOLOv3 pre-trained neural network. </h2>

```
# Defining classes for YOLO.

class_path = './yolov3.txt'
classes = None

with open(class_path, 'r') as f:
    classes = [line.strip() for line in f.readlines()]

COLORS = np.random.uniform(0, 255, size=(len(classes), 3))
scale = 0.00392

# Loading weights and configuration file

weights = 'yolov3.weights'
config = './yolov3.cfg'

# Loading the deep neural network on OpenCV

net = cv2.dnn.readNet(weights, config)

# Setting u the queues that will let the real time inference to happen via multithreading

frame_queue = Queue()
darknet_image_queue = Queue(maxsize=1)
detections_queue = Queue(maxsize=1)
fps_queue = Queue(maxsize=1)

```

<h2> Defining the functions that will perform inference </h2>

```
def get_output_layers(net):
    '''This function will get the output layers of YOLO pre-trained model'''
    layer_names = net.getLayerNames()
    try:
        output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    except:
        output_layers = [layer_names[i[0] - 1] for i in net.getUnconnectedOutLayers()]

    return output_layers

def draw_prediction(img, class_id, x, y, x_plus_w, y_plus_h):
    ''' This function will draw the prediction based on the inputs: image cropped, class_id and coordinates'''
    label = class_id
    if label == 'green':
        color = (0,255,0)
    elif label == 'red':
        color = (0,0,255)
    elif label == 'yellow':
        color = (255,255,0)

    cv2.rectangle(img, (x, y), (x_plus_w, y_plus_h), color, 1)

    cv2.putText(img, label, (x - 10, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

    return img

def video_capture(frame_queue, darknet_image_queue):
    '''OpenCV streaming function'''
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        #frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame, (416, 416),
                                   interpolation=cv2.INTER_LINEAR)
        frame_queue.put(frame_resized)
        darknet_image_queue.put(frame_resized)
    cap.release()

def inference(darknet_image_queue, detections_queue, fps_queue):
    ''' Inference function, takes the inputs from frames queue, process the current
    frame with loaded DNN, gets the output layers and returns the output prediction
    to the detections queue. Also makes the FPS calculation.
    '''
    while cap.isOpened():
        darknet_image = darknet_image_queue.get()
        blob = cv2.dnn.blobFromImage(darknet_image, scale, (416, 416), (0, 0, 0), True, crop=False)
        net.setInput(blob)
        prev_time = time.time()
        outs = net.forward(get_output_layers(net))
        class_ids = []
        confidences = []
        boxes = []
        conf_threshold = 0.95
        nms_threshold = 0.4

        for out in outs:
            for detection in out:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                if confidence > 0.5 and class_id == 9:
                    center_x = int(detection[0] * 416)
                    center_y = int(detection[1] * 416)
                    w = int(detection[2] * 416)
                    h = int(detection[3] * 416)
                    x = center_x - w / 2
                    y = center_y - h / 2
                    class_ids.append(class_id)
                    confidences.append(float(confidence))
                    boxes.append([x, y, w, h])

        indices = cv2.dnn.NMSBoxes(boxes, confidences, conf_threshold, nms_threshold)

        detections_queue.put([indices, boxes, class_ids, confidences])
        fps = int(1/(time.time() - prev_time))
        fps_queue.put(fps)
        print("FPS: {}".format(fps))
    cap.release()

def drawing(frame_queue, detections_queue, fps_queue):
    '''Gets the bbox prediction and the frame, normalize the images to 224x224
    and classifies them based on HSV spectrum.
    '''
    while cap.isOpened():
        frame = frame_queue.get()
        detections = detections_queue.get()
        fps = fps_queue.get()
        if frame is not None and len(detections[1]) > 0:
            image = frame
            for i in range(len(detections[1])):
                box = detections[1][i]
                class_id = detections[2][i]
                confidence = detections[3][i]
                x = box[0]
                y = box[1]
                w = box[2]
                h = box[3]

                x, y, w, h = int(x), int(y), int(w), int(h)
                try:
                    bbox_image = frame[y:y + h, x:x + w]
                    height, width = bbox_image.shape[:2]
                    medida = height * width
                    bbox_image = cv2.resize(bbox_image, (168, 224))
                    # pasar a hsv
                    hsv_img = cv2.cvtColor(bbox_image, cv2.COLOR_BGR2HSV)

                    # min and max HSV values for each color
                    red_min = np.array([0, 5, 150])
                    red_max = np.array([4, 255, 255])
                    red_min2 = np.array([175, 5, 150])
                    red_max2 = np.array([180, 255, 255])

                    yellow_min = np.array([25, 5, 150])
                    yellow_max = np.array([34, 255, 255])

                    green_min = np.array([40, 5, 150])
                    green_max = np.array([90, 255, 255])

                    red_thresh = cv2.inRange(hsv_img, red_min, red_max) + cv2.inRange(hsv_img, red_min2, red_max2)
                    yellow_thresh = cv2.inRange(hsv_img, yellow_min, yellow_max)
                    green_thresh = cv2.inRange(hsv_img, green_min, green_max)

                    # Apply blur to fix noise in thresh
                    red_blur = cv2.medianBlur(red_thresh, 5)
                    yellow_blur = cv2.medianBlur(yellow_thresh, 5)
                    green_blur = cv2.medianBlur(green_thresh, 5)

                    # Checks which colour thresh has the most white pixels
                    red = cv2.countNonZero(red_blur)
                    yellow = cv2.countNonZero(yellow_blur)
                    green = cv2.countNonZero(green_blur)

                    # The state of the light is the one with the greatest number of white pixels
                    lightColor = max(red, yellow, green)

                    if lightColor > 70:
                        if lightColor == red:
                            class_id = 'red'
                        elif lightColor == yellow:
                            class_id = 'yellow'
                        elif lightColor == green:
                            class_id = 'green'
                    
                    # A filter to prevent showing very little images, which are prone to errors.
                    
                    if medida>30:
                        image = draw_prediction(image, class_id, round(x), round(y), round(x + w), round(y + h))
                    cv2.imshow('Stream', image)
                    key = cv2.waitKey(1)
                    
                except:
                    raise ValueError('No frame detected.')
            if cv2.waitKey(fps) == 27:
                break
        elif frame is not None:
            cv2.imshow('Stream', frame)
            key = cv2.waitKey(1)
```
<h2> Perfoming inference </h2>

```
cap = cv2.VideoCapture('traffic_lights.avi')
video_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
Thread(target=video_capture, args=(frame_queue, darknet_image_queue)).start()
Thread(target=inference, args=(darknet_image_queue, detections_queue, fps_queue)).start()
Thread(target=drawing, args=(frame_queue, detections_queue, fps_queue)).start()

```

<h2>Output samples:</h2>


https://user-images.githubusercontent.com/105886661/216980237-429ccc79-10fa-43e4-b2a2-40cac7f281d3.mp4


https://user-images.githubusercontent.com/105886661/215539143-42164122-bb3a-4a87-9fb6-13c924155ea0.mp4


https://user-images.githubusercontent.com/105886661/215540790-e6274b7a-952c-495c-b779-a1b9f77f14a3.mp4


https://user-images.githubusercontent.com/105886661/215539295-606691b7-7c13-410c-ac8b-b2b9872c80c8.mp4





