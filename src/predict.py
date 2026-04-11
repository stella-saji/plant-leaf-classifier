import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
import os

CONFIDENCE_THRESHOLD = 0.7  # 70%

# 1. Load the trained model (SavedModel format)
model = tf.keras.models.load_model("plant_leaf_model")

# 2. Class names (IMPORTANT: order matters)
class_names = ['mango', 'neem', 'tulsi']

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
    
    # Get top prediction and confidence
    confidence = np.max(predictions[0])
    predicted_class = class_names[np.argmax(predictions[0])]
    
    # Get all predictions for display
    all_preds = predictions[0]

    print("\n" + "="*60)
    print(f"📸 Image: {img_name}")
    print("="*60)
    
    # Show confidence scores for all classes
    print("📊 Confidence Scores:")
    for i, class_name in enumerate(class_names):
        score = all_preds[i]
        bar = "█" * int(score * 30)  # Visual bar
        print(f"  {class_name.capitalize():10} {score*100:6.2f}% {bar}")
    
    print(f"\n🔝 Top Prediction: {predicted_class.capitalize()} ({confidence*100:.2f}%)")

    if confidence < CONFIDENCE_THRESHOLD:
        print("⚠️  Confidence below threshold - May be an unknown/unsupported leaf")
    else:
        print("✅ High confidence prediction!")
        print("\n💊 Uses & Benefits:")
        for use in plant_info[predicted_class]["uses"]:
           print(f"  • {use}")

print("\n" + "="*60)


