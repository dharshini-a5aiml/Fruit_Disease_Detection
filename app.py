import os
import json
import uuid
import numpy as np
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from PIL import Image
import tensorflow as tf

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "model", "fruit_model.keras")
CLASS_PATH = os.path.join(BASE_DIR, "model", "class_names.json")
ALLOWED = {"jpg", "jpeg", "png"}

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB limit

# Load the trained model once, when the app starts
model = None
class_names = []
load_error = None
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(CLASS_PATH) as f:
        class_names = json.load(f)
except Exception as e:
    load_error = ("Model files not found. Please run train.py first "
                  "so that model/fruit_model.keras is created.")
    print("Model load error:", e)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED


def pretty_label(name):
    # "Blotch_Apple" -> ("Apple", "Blotch")
    disease, fruit = name.rstrip("_").rsplit("_", 1)
    disease = disease.strip("_").replace("_", " ")
    return fruit, disease


def predict_image(path):
    img = Image.open(path).convert("RGB").resize((224, 224))
    arr = np.expand_dims(np.array(img, dtype="float32"), axis=0)
    # no /255 here: the model already rescales inside itself
    probs = model.predict(arr, verbose=0)[0]
    idx = int(np.argmax(probs))
    return class_names[idx], float(probs[idx]) * 100


@app.errorhandler(413)
def too_large(e):
    return render_template("index.html", error="File too large. Max size is 8 MB."), 413


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "GET":
        return render_template("index.html", error=load_error)

    if model is None:
        return render_template("index.html", error=load_error)

    file = request.files.get("image")
    if file is None or file.filename == "":
        return render_template("index.html", error="Please choose an image first.")
    if not allowed_file(file.filename):
        return render_template("index.html",
                               error="Invalid file type. Upload a JPG or PNG image.")

    ext = secure_filename(file.filename).rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    path = os.path.join(UPLOAD_FOLDER, filename)
    file.save(path)

    try:
        label, confidence = predict_image(path)
    except Exception:
        os.remove(path)
        return render_template("index.html",
                               error="Could not read this image. Try another file.")

    fruit, disease = pretty_label(label)
    return render_template(
        "index.html",
        image_file="uploads/" + filename,
        fruit=fruit,
        disease=disease,
        confidence=round(confidence, 1),
        low_confidence=confidence < 60,
    )


if __name__ == "__main__":
    app.run(debug=False)