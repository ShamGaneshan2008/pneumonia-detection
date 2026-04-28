from flask import Flask, request, jsonify, send_from_directory
from fastai.vision.all import *
import os
import cv2
from werkzeug.utils import secure_filename
from utils import generate_gradcam

app = Flask(__name__, static_folder='../static', template_folder='../app/templates')
app.secret_key = 'change-this-to-a-random-string'

UPLOAD_FOLDER = '../static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
MAX_FILE_SIZE_MB = 5

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = MAX_FILE_SIZE_MB * 1024 * 1024

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Load the model once when the server starts, not on every request
try:
    learn = load_learner('../models/model.pkl')
except Exception as e:
    learn = None
    print(f"Model failed to load: {e}")


def allowed_file(filename):
    # Check the extension, not just that a dot exists
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def home():
    return send_from_directory('../app/templates', 'index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if learn is None:
        return jsonify(error='Model unavailable, contact the admin.'), 503

    if 'file' not in request.files or request.files['file'].filename == '':
        return jsonify(error='No file selected.'), 400

    file = request.files['file']

    if not allowed_file(file.filename):
        return jsonify(error=f'Unsupported file type. Use: {", ".join(ALLOWED_EXTENSIONS)}'), 400

    # secure_filename strips things like "../" that could overwrite files outside the upload folder
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        pred, idx, probs = learn.predict(filepath)
        confidence = round(float(probs[idx]) * 100, 2)
    except Exception as e:
        return jsonify(error=f'Prediction failed: {e}'), 500

    # Heatmap is optional — if it fails we still return the prediction
    heatmap_filename = None
    try:
        heatmap = generate_gradcam(learn, filepath)
        heatmap_filename = 'heatmap_' + filename
        cv2.imwrite(os.path.join(app.config['UPLOAD_FOLDER'], heatmap_filename), heatmap)
    except Exception as e:
        print(f"Grad-CAM skipped: {e}")

    return jsonify(
        prediction=str(pred),
        confidence=confidence,
        image=filename,
        heatmap=heatmap_filename  # None if heatmap failed, JS handles that
    )


@app.errorhandler(413)
def file_too_large(e):
    return jsonify(error=f'File exceeds the {MAX_FILE_SIZE_MB} MB limit.'), 413


if __name__ == '__main__':
    app.run(debug=True)