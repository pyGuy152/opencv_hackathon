from waitress import serve
from flask import Flask, request, jsonify
from ultralytics import YOLO
import cv2
cv2.setUseOptimized(True)

import numpy as np

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


app = Flask(__name__)

@app.route("/ai")
def ai():
    return (f'{x} {y} {w} {h}')

@app.route('/api', methods=['POST'])
def api():
    try:
        data = request.get_json()  # Attempt to parse JSON data
        if data is None:
            return jsonify({"error": "Invalid JSON data"}), 400 #Return an error if not valid JSON
        return jsonify({"message": "API POST request received", "data": data})

    except Exception as e:
        return jsonify({"error": str(e)}), 400 #Return an error if the request is not valid.

if __name__ == "__main__":
    serve(app, host='0.0.0.0', port=8000)