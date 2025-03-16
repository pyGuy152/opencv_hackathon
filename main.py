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


# Define where to save the uploaded data (you can change this as needed)
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

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
        # Save the file to a folder on the server
        filename = os.path.join(UPLOAD_FOLDER, 'uploaded_image.jpg')
        file.save(filename)
        
        # Read the image (you can do further processing here)
        img = Image.open(filename)
        
        # Example: display image info (you can add your processing logic here)
        print("Image dimensions:", img.size)
        print("Image format:", img.format)
        
        # Perform any other image processing (e.g., resizing, analyzing, etc.)
        
        return f"Image uploaded successfully and saved as {filename}", 200
    else:
        return "File not received", 400

if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=8000)
