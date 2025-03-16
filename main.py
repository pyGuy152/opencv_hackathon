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
@app.route('/upload_stream', methods=['POST'])
def upload_stream():
    # Check if the request contains the stream data
    if 'file' not in request.files:
        return "No file part", 400

    # Retrieve the file from the request
    file = request.files['file']
    
    # Save the file to the upload folder (or process it as needed)
    if file:
        img = cv2.imread(file)

        # Convert BGR to RGB for displaying with matplotlib
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        # Perform object detection on the image
        results = model(img_rgb)

        # Get results in format (xyxy, confidence, class) for detected objects
        predictions = results[0].boxes  # Detected boxes
        x, y, w, h = map(int, predictions.xywh[0])
        return f"{x} {y} {w} {h}"

    return "File not received", 400

@app.route("/ai")
def ai():
    return f'{x} {y} {w} {h}'

@app.route("/stream")
def ai():
    return f'{x} {y} {w} {h}'


@app.route('/api', methods=['POST'])
def api():
    try:
        data = request.get_json()
        if data is None:
            return jsonify({"error": "Invalid JSON data"}), 400
        return jsonify({"message": "API POST request received", "data": data})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=8000)
