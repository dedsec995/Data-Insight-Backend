from flask import Flask, request, jsonify, session
from flask_cors import CORS
import os, time, random, uuid, base64
from io import BytesIO

app = Flask(__name__)
app.secret_key = 'mySecret'
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

    return (
        jsonify(
            {
                "message": "File uploaded successfully",
                "session_id": session_id,
            }
        ),
        200,
    )


@app.route("/data/<session_id>", methods=["GET"])
def get_graph_data(session_id):
    image_paths = [
        "/home/dedsec995/datainsight/new_try/backend/samples/corr.png",
        "/home/dedsec995/datainsight/new_try/backend/samples/corr.png",
    ]

    images = []

    for image_path in image_paths:
        with open(image_path, "rb") as image_file:
            encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
            images.append(encoded_string)

    return jsonify({"images": images})


if __name__ == '__main__':
    app.run(debug=True)
