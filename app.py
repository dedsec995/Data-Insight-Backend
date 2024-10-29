from flask import Flask, request, jsonify, session, send_file
from flask_cors import CORS
import os, time, random, uuid, base64, json
from io import BytesIO
from graphs import combined_visualizations
from suggestions import suggestions
from model_creation import create_model
import pickle, re

app = Flask(__name__)
app.secret_key = "mySecret"
CORS(app, supports_credentials=True)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

dummy_data = {
    "model_path": "/home/dedsec995/datainsight/backend/playground/model/loan_decision_tree_model.pkl",
    "conf_path": "/home/dedsec995/datainsight/backend/playground/model/loan_confusion_matrix.png",
    "result": {
        "accuracy": 0.89505,
        "recoil": 0.89505,
        "precision": 0.8982837829034723,
        "f1": 0.8965567235596356,
        "support": 20000.0,
    },
}

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
        all_image_paths = []
        if isinstance(corr_result, str):
            all_image_paths.append(corr_result)
        elif isinstance(corr_result, list):
            all_image_paths.extend(corr_result)
        if isinstance(hist_result, list):
            all_image_paths.extend(hist_result)
        elif isinstance(hist_result, str):
            all_image_paths.append(hist_result)
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


@app.route("/api/suggestions/<session_id>", methods=["GET"])
def get_suggestions(session_id):
    if not session_id:
        return jsonify({"error": "No session ID provided"}), 400

    folder_path = os.path.join(UPLOAD_FOLDER, session_id)
    if not os.path.exists(folder_path):
        return jsonify({"error": "Invalid session ID"}), 400

    file_path = None
    for file in os.listdir(folder_path):
        if file.lower().endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            break

    if not file_path:
        return jsonify({"error": "No CSV file found for this session"}), 400

    try:
        result = suggestions(file_path)
        return jsonify(result)

    except Exception as e:
        return jsonify({"error": f"An error occurred: {str(e)}"}), 500


@app.route("/api/result/<session_id>", methods=["POST"])
def get_result(session_id):
    # In a real application, you might use the session_id and suggestion
    # to retrieve or process data. For this example, we'll ignore them.
    suggestion = request.json["suggestion"]
    folder_path = os.path.join(UPLOAD_FOLDER, session_id)

    if not os.path.exists(folder_path):
        return jsonify({"error": "Invalid session ID"}), 400

    file_path = None
    for file in os.listdir(folder_path):
        if file.lower().endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            break

    result = create_model(file_path,folder_path,suggestion)

    if result is None:
        return jsonify({"error": "Result is None"}), 400
    json_match = re.search(r"\{.*\}", result, re.DOTALL)
    if not json_match:
        return jsonify({"error": "No JSON data found in the output"}), 400

    try:
        data = json.loads(json_match.group(0))
    except json.JSONDecodeError:
        return jsonify({"error": "Invalid JSON data in the output"}), 400
    try:
        conf_path = data.get('conf_path')
    except:
        print("Conf path not found")
        conf_path = None
    try:
        with open(conf_path, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
    except FileNotFoundError:
        print(f"Confusion matrix image not found at {conf_path}"),
    except Exception as e:
        print(f"Error reading confusion matrix image: {str(e)}")

    response_data = {
        "model_path": data.get("model_path"),
        "conf_image": encoded_image,
        "result": data.get("result"),
    }
    return jsonify(response_data)


@app.route("/api/download_model/<session_id>", methods=["GET"])
def download_model(session_id):
    # In a real application, you might use the session_id to authorize the download
    return send_file(dummy_data["model_path"], as_attachment=True)


if __name__ == '__main__':
    app.run(debug=False)
