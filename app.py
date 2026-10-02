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

DISEASE_INFO = {
    "Apple|Blotch": ("A fungal disease that causes dark blotches on the skin of the fruit.",
                     "Remove infected fruit, prune for better air flow, and ask a local agriculture officer about a suitable fungicide."),
    "Apple|Rot": ("Fruit decay that appears as soft, brown, sunken patches.",
                  "Remove and dispose of rotten fruit, avoid bruising during handling, and store fruit in a cool, dry place."),
    "Apple|Scab": ("A fungal disease that causes dark, rough, scab-like spots on fruit and leaves.",
                   "Clear fallen leaves and fruit, prune the tree for air flow, and follow local advice on fungicide timing."),
    "Apple|Healthy": ("No visible disease detected on this apple.",
                      "Keep up regular care: balanced watering, pruning and checking the fruit often."),
    "Guava|Anthracnose": ("A fungal disease that causes dark, sunken spots on the fruit.",
                          "Remove infected fruit, avoid injuring fruit during harvest, and keep the area around the tree clean."),
    "Guava|Fruitfly": ("Damage caused by fruit fly larvae that feed inside the fruit.",
                       "Collect and destroy fallen fruit, use fruit fly traps, and bag young fruits to protect them."),
    "Guava|Healthy": ("No visible disease detected on this guava.",
                      "Keep up regular care and check the fruit often for early signs of damage."),
    "Mango|Alternaria": ("A fungal disease that causes dark spots and rot on the fruit surface.",
                         "Handle fruit gently, remove infected fruit, and keep harvested fruit dry and well ventilated."),
    "Mango|Anthracnose": ("A fungal disease that causes black, sunken spots, especially as fruit ripens.",
                          "Remove infected fruit, prune for air flow, and handle fruit carefully after harvest."),
    "Mango|Black Mould Rot Aspergillus": ("A fungal rot that produces black, powdery mould on the fruit.",
                                          "Discard infected fruit, avoid wounds during harvest, and store fruit in clean, dry conditions."),
    "Mango|Stem and Rot Lasiodiplodia": ("A fungal rot that usually starts at the stem end of the fruit.",
                                         "Avoid injuring the stem during harvest, remove infected fruit, and keep storage clean and dry."),
    "Mango|Healthy": ("No visible disease detected on this mango.",
                      "Keep up regular care and check the fruit often for early signs of damage."),
}


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

    # take the 3 highest probabilities
    top3 = np.argsort(probs)[::-1][:3]
    results = []
    for i in top3:
        fruit, disease = pretty_label(class_names[int(i)])
        results.append({
            "fruit": fruit,
            "disease": disease,
            "confidence": round(float(probs[i]) * 100, 1),
        })
    return results


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
        results = predict_image(path)
    except Exception:
        os.remove(path)
        return render_template("index.html",
                               error="Could not read this image. Try another file.")

    best = results[0]
    info = DISEASE_INFO.get(best["fruit"] + "|" + best["disease"])
    description = info[0] if info else None
    care_tip = info[1] if info else None
    return render_template(
        "index.html",
        image_file="uploads/" + filename,
        fruit=best["fruit"],
        disease=best["disease"],
        confidence=best["confidence"],
        low_confidence=best["confidence"] < 60,
        top3=results,
        description=description,
        care_tip=care_tip,
    )


if __name__ == "__main__":
    import threading
    import webbrowser

    # open the browser 1.5 seconds after the server starts
    threading.Timer(1.5, lambda: webbrowser.open("http://127.0.0.1:5000")).start()
    app.run(debug=False)