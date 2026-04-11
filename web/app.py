from flask import Flask, render_template, request, flash
import tensorflow as tf
import numpy as np
import os
import secrets
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename
from uuid import uuid4

# -------------------- APP SETUP --------------------
app = Flask(__name__)

# Load SECRET_KEY from environment or generate secure random key for development
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))

# Warning for development mode
if 'SECRET_KEY' not in os.environ:
    print("⚠️  Using auto-generated development SECRET_KEY")
    print("   For production, set environment variable: SECRET_KEY='your-secure-key'")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
MODEL_PATHS = [
    os.path.join(BASE_DIR, "..", "plant_leaf_model.keras"),
    os.path.join(BASE_DIR, "..", "plant_leaf_model.h5"),
]

ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
CONFIDENCE_THRESHOLD = 0.7

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024  # 5 MB limit

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# -------------------- LOAD MODEL --------------------
model = None
for model_path in MODEL_PATHS:
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        break

if model is None:
    raise FileNotFoundError(
        "No model file found. Expected one of: "
        + ", ".join(MODEL_PATHS)
    )

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
    return render_template("app.html", show_predict=False)


@app.route("/predict", methods=["GET", "POST"])
def predict():
    prediction = None
    confidence = None
    all_confidences = None
    uses = None
    image_name = None

    if request.method == "POST":
        file = request.files.get("image")

        if not file or not file.filename:
            flash("❌ Please select a file to upload", "error")
        elif not allowed_file(file.filename):
            flash("❌ Invalid file type. Please upload PNG, JPG, or JPEG", "error")
        else:
            safe_name = secure_filename(file.filename)
            unique_name = f"{uuid4().hex}_{safe_name}"
            img_path = os.path.join(app.config["UPLOAD_FOLDER"], unique_name)
            file.save(img_path)

            img = image.load_img(img_path, target_size=(224, 224))
            img_array = image.img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            preds = model.predict(img_array)[0]
            confidence = float(np.max(preds))
            predicted_class = class_names[np.argmax(preds)]
            
            # Store all confidences for confidence bars
            all_confidences = [
                {"class": class_names[i].capitalize(), "score": float(preds[i])}
                for i in range(len(class_names))
            ]

            if confidence >= CONFIDENCE_THRESHOLD:
                prediction = predicted_class.capitalize()
                uses = plant_info[predicted_class]
            else:
                prediction = "Unknown or unsupported leaf"
                uses = None

            image_name = unique_name

    return render_template(
        "app.html",
        show_predict=True,
        prediction=prediction,
        confidence=confidence,
        all_confidences=all_confidences,
        uses=uses,
        image_name=image_name
    )


# -------------------- RUN --------------------
if __name__ == "__main__":
    # Run in development mode (debug=True only for local development)
    debug_mode = os.environ.get('FLASK_ENV') == 'development'
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=debug_mode, host='0.0.0.0', port=port)
