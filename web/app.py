from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from uuid import uuid4

# -------------------- APP SETUP --------------------
app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
MODEL_PATH = os.path.join(BASE_DIR, "..", "plant_leaf_model.h5")

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
CONFIDENCE_THRESHOLD = 0.7

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB limit

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------- LOAD MODEL --------------------
model = tf.keras.models.load_model(MODEL_PATH)

class_names = ["mango", "neem", "tulsi"]

plant_info = {
    "neem": [
        "Antibacterial and antifungal",
        "Used in skin care",
        "Boosts immunity",
        "Dental care (neem sticks)"
    ],
    "mango": [
        "Improves digestion",
        "Rich in vitamins A and C",
        "Boosts immunity",
        "Used in traditional medicine"
    ],
    "tulsi": [
        "Relieves cold and cough",
        "Reduces stress",
        "Improves respiratory health",
        "Sacred medicinal herb in India"
    ]
}

# -------------------- HELPERS --------------------
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# -------------------- ROUTES --------------------
@app.route("/")
def home():
    return render_template("home.html")


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    confidence = None
    uses = None
    image_name = None

    if request.method == "POST":
        file = request.files.get("image")

        if file and file.filename and allowed_file(file.filename):
            safe_name = secure_filename(file.filename)
            unique_name = f"{uuid4().hex}_{safe_name}"
            img_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
            file.save(img_path)

            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array)
            confidence = float(np.max(preds))
            predicted_class = class_names[np.argmax(preds)]

            if confidence >= CONFIDENCE_THRESHOLD:
                prediction = predicted_class.capitalize()
                uses = plant_info[predicted_class]
            else:
                prediction = "Unknown or unsupported leaf"

            image_name = unique_name

        else:
            prediction = "Invalid file type. Please upload a leaf image."

    return render_template(
        "predict.html",
        prediction=prediction,
        confidence=confidence,
        uses=uses,
        image_name=image_name
    )


# -------------------- RUN --------------------
if __name__ == "__main__":
    app.run(debug=True)
