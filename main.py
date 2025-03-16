from waitress import serve
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import numpy as np
import torch
import cv2
import os

# Load the pre-trained YOLOv8 model (or you can use a smaller model like 'yolov5')
model = YOLO('yolov8n.pt')  # You can replace with 'yolov8s.pt', 'yolov8m.pt', etc.

# Load the image using OpenCV
img_path = 'premium_photo-1674170065323-9f207919ea27.jpeg'
img = cv2.imread(img_path)

# Convert BGR to RGB for displaying with matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Perform object detection on the image
results = model(img_rgb)

# Get results in format (xyxy, confidence, class) for detected objects
predictions = results[0].boxes  # Detected boxes
x, y, w, h = map(int, predictions.xywh[0])
print(x,y,w,h)

if len(predictions) > 0:
    x, y, w, h = map(int, predictions.xywh[0])
else:
    x, y, w, h = 0, 0, 0, 0

app = Flask(__name__)



# Define the route for uploading a stream of data

@app.route("/ai")
def ai():
    return f'{x} {y} {w} {h}'

@app.route('/api', methods=['POST'])
def api():
    # Check if the request contains the file data
    if 'file' not in request.files:
        return "No file part", 400
    
    # Retrieve the file
    file = request.files['file']
    
    if file:
        image = Image.open(file)
        model = YOLO('yolov8n.pt')
        img = cv2.imread(file)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = model(img_rgb)
        predictions = results[0].boxes
        x, y, w, h = map(int, predictions.xywh[0])
        width, height = image.size
        center = [x+(w/2),y+(h/2)]
        if (center < width/2-10):
            return "Left", 200
        elif (center > width/2+10):
            return "Right", 200
        elif ((x*h)/(width*height) > 0.75):
            return "Stop", 200
        else:
            return "Forward", 200
    else:
        return "File not received", 400

if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=8000)
