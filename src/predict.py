import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image

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

predictions = model.predict(img_array)
confidence = np.max(predictions)
predicted_class = class_names[np.argmax(predictions)]

# 5. Output result
print("🌿 Predicted Plant:", predicted_class.capitalize())
print("📊 Confidence:", round(confidence * 100, 2), "%")

print("\n💊 Medicinal / Common Uses:")
for use in plant_info[predicted_class]["uses"]:
    print("-", use)

