from waitress import serve
from flask import Flask, request, jsonify

app = Flask(__name__)

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