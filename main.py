from flask import Flask, request, jsonify
from ultralytics import YOLO
from PIL import Image
import numpy as np
import torch
import cv2
import os
import traceback  # To print the traceback in case of errors

# Load the model
model = YOLO('yolov8n.pt')

app = Flask(__name__)


# Set upload folder
UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route('/api', methods=['POST'])
def api():
    try:
        # Check if the request contains the file data
        if 'file' not in request.files:
            return "No file part", 400

        file = request.files['file']

        if file:
            #if file.filename == '':
            #    return 'No selected file'
            
            #filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            #file.save(filepath)
            # Open the image using PIL and then process it
            image = Image.open(file)
            img = np.array(image)
            # os.remove(filepath)
            if img is None:
                return "Error: Could not read image", 400
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            results = model(img_rgb)
            predictions = results[0].boxes
            x, y, w, h = map(int, predictions.xywh[0])

            width, height = image.size
            center = [x + (w / 2), y + (h / 2)]

            if center[0] < width / 2 - 50:
                return "Left", 200
            elif center[0] > width / 2 + 50:
                return "Right", 200
            elif (x * h) / (width * height) > 0.75:
                return "Stop", 200
            else:
                return "Forward", 200
        else:
            return "File not received", 400
    except Exception as e:
        # Log the error
        error_message = str(e)
        traceback_str = traceback.format_exc()
        print(f"Error occurred: {error_message}")
        print(traceback_str)
        return f"Internal Server Error: {error_message}", 500

if __name__ == "__main__":
    from waitress import serve
    serve(app, host='0.0.0.0', port=8000)
