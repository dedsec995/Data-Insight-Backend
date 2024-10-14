from flask import Flask, request, jsonify, session
from flask_cors import CORS
import os, time, random, uuid, base64
from io import BytesIO
from graphs import combined_visualizations

app = Flask(__name__)
app.secret_key = "mySecret"
CORS(app, supports_credentials=True)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)


@app.route("/upload", methods=["POST"])
def upload_file():
    if "file" not in request.files:
        return jsonify({"error": "No file part"}), 400
    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "No selected file"}), 400

    # Create a unique session ID and folder for this upload
    session_id = str(uuid.uuid4())
    session['session_id'] = session_id
    folder_path = os.path.join(UPLOAD_FOLDER, session_id)
    os.makedirs(folder_path, exist_ok=True)

    # Save the file
    filepath = os.path.join(folder_path, file.filename)
    file.save(filepath)

    try:
        corr_result, hist_result = combined_visualizations(filepath, folder_path)
        all_image_paths = (
            corr_result + hist_result if isinstance(corr_result, list) else hist_result
        )
        session['image_paths'] = all_image_paths
        return (
            jsonify(
                {
                    "message": "File uploaded and processed successfully",
                    "session_id": session_id,
                    "image_paths": all_image_paths,
                }
            ),
            200,
        )
    except Exception as e:
        return jsonify({"error": f"An error occurred during processing: {str(e)}"}), 500


@app.route("/data/<session_id>", methods=["POST"])
def get_images(session_id):
    image_paths = request.json.get("imagePaths", [])

    encoded_images = []

    for image_path in image_paths:
        if os.path.exists(image_path):
            with open(image_path, "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read()).decode("utf-8")
                encoded_images.append(encoded_string)
        else:
            print(f"Image not found: {image_path}")

    return jsonify({"images": encoded_images}), 200


if __name__ == '__main__':
    app.run(debug=True)
