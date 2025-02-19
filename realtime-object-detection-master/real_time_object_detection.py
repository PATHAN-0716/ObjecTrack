import cv2
import numpy as np
import argparse
import time
from imutils.video import VideoStream, FPS
import imutils

# Argument parser for model paths
parser = argparse.ArgumentParser()
parser.add_argument("-p", "--prototxt", required=True, help="Path to the Caffe deploy prototxt file")
parser.add_argument("-m", "--model", required=True, help="Path to the Caffe pre-trained model")
parser.add_argument("-t", "--threshold", type=float, default=0.2, help="Confidence threshold for filtering detections")
args = vars(parser.parse_args())

# List of object categories the model can detect
OBJECTS = ["background", "bicycle", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "phone", "tvmonitor"]

# Generate distinct colors for each category
BOX_COLORS = np.random.uniform(0, 255, size=(len(OBJECTS), 3))

# Load the trained model
print("[INFO] Loading object detection model...")
detector = cv2.dnn.readNetFromCaffe(args["prototxt"], args["model"])

# Start the video feed
print("[INFO] Starting camera stream...")
video = VideoStream(src=0).start()
time.sleep(2.0)
fps_counter = FPS().start()

# Process video frames
while True:
    # Capture and resize frame
    frame = video.read()
    frame = imutils.resize(frame, width=500)
    (height, width) = frame.shape[:2]
    
    # Convert image into a format suitable for the model
    image_blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
    detector.setInput(image_blob)
    predictions = detector.forward()
    
    # Iterate over detected objects
    for i in range(predictions.shape[2]):
        confidence = predictions[0, 0, i, 2]
        
        # Only process objects with confidence above threshold
        if confidence > args["threshold"]:
            obj_id = int(predictions[0, 0, i, 1])
            bbox = predictions[0, 0, i, 3:7] * np.array([width, height, width, height])
            (x1, y1, x2, y2) = bbox.astype("int")
            
            # Draw bounding box and label
            label_text = "{}: {:.2f}%".format(OBJECTS[obj_id], confidence * 100)
            cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLORS[obj_id], 2)
            label_position = y1 - 10 if y1 - 10 > 10 else y1 + 10
            cv2.putText(frame, label_text, (x1, label_position), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BOX_COLORS[obj_id], 2)
    
    # Display processed frame
    cv2.imshow("Object Detection", frame)
    key = cv2.waitKey(1) & 0xFF
    
    # Exit when 'q' key is pressed
    if key == ord("q"):
        break
    
    fps_counter.update()

# Display performance metrics
fps_counter.stop()
print("[INFO] Total runtime: {:.2f} seconds".format(fps_counter.elapsed()))
print("[INFO] Estimated FPS: {:.2f}".format(fps_counter.fps()))

# Cleanup
cv2.destroyAllWindows()
video.stop()

#to run
#python real_time_object_detection.py --prototxt MobileNetSSD_deploy.prototxt.txt --model MobileNetSSD_deploy.caffemodel
#cd "E:\Projects\object detection\realtime-object-detection-master"
