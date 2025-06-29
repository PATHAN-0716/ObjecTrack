# 🧠 Real-Time Object Detection using MobileNet-SSD and OpenCV

A computer vision project that uses the **MobileNet-SSD** deep learning model with **OpenCV** to detect objects in real-time via a webcam or video file.

---

## 🚀 Project Overview

This project implements a lightweight and fast object detection system using **MobileNet-SSD**, ideal for edge devices or low-resource environments. It detects 20 common object classes using OpenCV's DNN module.

---

## 🧰 Features

- Real-time object detection with bounding boxes and labels
- MobileNet-SSD model loaded via OpenCV
- Supports both webcam and video input
- Fast and lightweight
- Color-coded object classes
- FPS counter for performance evaluation

---

## 🖼 Object Classes Detected

The MobileNet-SSD model detects the following object classes:

```
['background', 'aeroplane', 'bicycle', 'bird', 'boat',
 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable',
 'dog', 'horse', 'motorbike', 'person', 'pottedplant',
 'sheep', 'sofa', 'train', 'tvmonitor']
```

---

## 📁 Project Structure

```
📦mobilenet-ssd-object-detection/
 ┣ 📂models/
 ┃ ┣ MobileNetSSD_deploy.caffemodel
 ┃ ┗ MobileNetSSD_deploy.prototxt
 ┣ 📂videos/
 ┃ ┗ sample_video.mp4
 ┣ object_detection.py
 ┣ README.md
```

---

## 🔧 Requirements

Install dependencies using pip:

```bash
pip install opencv-python numpy
```

---

## 📥 Model Files

Download the following files and place them in the `models/` directory:

- [MobileNetSSD_deploy.caffemodel](https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.caffemodel)
- [MobileNetSSD_deploy.prototxt](https://github.com/chuanqi305/MobileNet-SSD/raw/master/MobileNetSSD_deploy.prototxt)

---

## ▶️ Running the Code

### From Webcam:

```bash
python object_detection.py --source webcam
```

### From Video File:

```bash
python object_detection.py --source videos/sample_video.mp4
```

---

## 🧠 How It Works

- Loads MobileNet-SSD using OpenCV's `cv2.dnn.readNetFromCaffe()`
- Reads frames from video or webcam
- Passes each frame through the network
- Draws bounding boxes and labels on detected objects
- Displays FPS in real time

---

## 🧪 Example Code Snippet

```python
net = cv2.dnn.readNetFromCaffe(prototxt_path, model_path)
blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
net.setInput(blob)
detections = net.forward()
```

---

## 💡 Future Improvements

- Add tracking with OpenCV or Deep SORT
- Deploy using Flask for web interface
- Add alert system for specific object detection
- Use PiCamera or Jetson Nano for edge deployment

---

## 🙋‍♂️ Author

**PATHAN ADILSHA KHAN**  
[GitHub](https://github.com/PATHAN-0716) | [LinkedIn](https://www.linkedin.com/in/pathan-adilsha-khan-1a6840259/)
