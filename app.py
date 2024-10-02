from flask import Flask, request, jsonify, session
from flask_cors import CORS
import os
import time
import random
import uuid

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Required for session management
CORS(app)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    # Create a unique session ID and folder for this upload
    session_id = str(uuid.uuid4())
    session['session_id'] = session_id
    folder_path = os.path.join(UPLOAD_FOLDER, session_id)
    os.makedirs(folder_path, exist_ok=True)

    # Save the file
    filepath = os.path.join(folder_path, file.filename)
    file.save(filepath)

    # Simulate processing time
    time.sleep(3)  # Simulate a delay for processing

    # Mock processing result (e.g., data for graphs)
    processed_data = {
        "labels": ["A", "B", "C", "D"],
        "values": [random.randint(1, 100) for _ in range(4)]
    }
    return (
        jsonify(
            {
                "message": "File uploaded successfully",
                "session_id": session_id,  # Include session ID in the response
                "data": processed_data,
            }
        ),
        200,
    )

@app.route('/data/<session_id>', methods=['GET'])
def get_graph_data(session_id):
    # Here you would retrieve the processed data associated with the session_id
    # This is a mock; implement your actual logic to retrieve the data
    processed_data = {
        "labels": ["A", "B", "C", "D"],
        "values": [random.randint(1, 100) for _ in range(4)]
    }
    return jsonify({'data': processed_data}), 200


if __name__ == '__main__':
    app.run(debug=True)
