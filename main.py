from waitress import serve
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import numpy as np
import torch
import cv2

# Load the pre-trained YOLOv8 model (or you can use a smaller model like 'yolov5')
model = YOLO('yolov8n.pt')  # You can replace with 'yolov8s.pt', 'yolov8m.pt', etc.

# Load the image using OpenCV
img_path = '/kaggle/input/jjjjjjjjj/gettyimages-1462659206-612x612.jpg'
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

@app.route("/ai")
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
