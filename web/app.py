from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import os
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER


# Load model once (important!)
model = tf.keras.models.load_model("plant_leaf_model.h5")

class_names = ['mango', 'neem', 'tulsi']
CONFIDENCE_THRESHOLD = 0.7

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

        if file:
            image_name = secure_filename(file.filename)
            img_path = os.path.join(app.config["UPLOAD_FOLDER"], image_name)
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

    # ✅ ALWAYS return a response (GET or POST)
    return render_template(
        "predict.html",
        prediction=prediction,
        confidence=confidence,
        uses=uses,
        image_name=image_name
    )

if __name__ == "__main__":
    app.run(debug=True)
