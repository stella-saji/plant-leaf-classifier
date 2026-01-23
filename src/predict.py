import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os
CONFIDENCE_THRESHOLD = 0.7  # 70%

# 1. Load the trained model
model = tf.keras.models.load_model("plant_leaf_model.h5")

# 2. Class names (IMPORTANT: order matters)
class_names = ['mango', 'neem', 'tulsi']

# 3. Load and preprocess the image
img_path = "test_images/test_leaf.jpg"

img = image.load_img(img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = img_array / 255.0
img_array = np.expand_dims(img_array, axis=0)

# 4. Make prediction
# Medicinal / use information
plant_info = {
    "neem": {
        "uses": [
            "Antibacterial and antifungal",
            "Used in skin care",
            "Boosts immunity",
            "Dental care (neem sticks)"
        ]
    },
    "mango": {
        "uses": [
            "Improves digestion",
            "Rich in vitamins A and C",
            "Boosts immunity",
            "Used in traditional medicine"
        ]
    },
    "tulsi": {
        "uses": [
            "Relieves cold and cough",
            "Reduces stress",
            "Improves respiratory health",
            "Sacred medicinal herb in India"
        ]
    }
}

test_dir = "test_images"
image_files = [f for f in os.listdir(test_dir) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]

if not image_files:
    raise FileNotFoundError("❌ No images found in test_images folder")

for img_name in image_files:
    img_path = os.path.join(test_dir, img_name)

    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(img_array)
    confidence = np.max(predictions)
    predicted_class = class_names[np.argmax(predictions)]

    

# 5. Output result
    print("\n📸 Image:", img_name)
    print("📊 Confidence:", round(confidence * 100, 2), "%")

    if confidence < CONFIDENCE_THRESHOLD:
        print("⚠️ Unknown or unsupported leaf image")
    else:
        print("🌿 Predicted Plant:", predicted_class.capitalize())
        print("💊 Uses:")
        for use in plant_info[predicted_class]["uses"]:
           print("-", use)


