from waitress import serve
from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import numpy as np

# Load the pre-trained YOLOv8 model
model = YOLO('yolov8n.pt')

# Load the image using PIL instead of OpenCV
img_path = 'premium_photo-1674170065323-9f207919ea27.jpeg'
img = Image.open(img_path).convert('RGB')  # Ensure RGB format

# Convert image to NumPy array
img_np = np.array(img)

# Perform object detection
results = model(img_np)

# Get bounding box results
predictions = results[0].boxes
x, y, w, h = map(int, predictions.xywh[0])

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
