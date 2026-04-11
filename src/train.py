import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
import os

# ==================== IMAGE SETTINGS ====================
IMAGE_SIZE = (224, 224)
BATCH_SIZE = 8

# ==================== DATA AUGMENTATION ====================
# Initial training with strong augmentation to reduce overfitting
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    horizontal_flip=True,
    zoom_range=0.2,
    brightness_range=[0.8, 1.2],
    shear_range=0.1,
    fill_mode='nearest',
    validation_split=0.2
)

# Validation data: only rescaling (no augmentation)
val_datagen = ImageDataGenerator(rescale=1./255)

# ==================== LOAD DATA ====================
print("📂 Loading training data...")
train_data = train_datagen.flow_from_directory(
    "dataset/train",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training"
)

val_data = train_datagen.flow_from_directory(
    "dataset/train",
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation"
)

print(f"✅ Training samples: {len(train_data)}, Validation samples: {len(val_data)}")

# ==================== PHASE 1: TRAIN WITH FROZEN BASE ====================
print("\n🔒 Phase 1: Training with frozen base model...")

# Load pre-trained MobileNetV2
base_model = MobileNetV2(
    weights="imagenet",
    include_top=False,
    input_shape=(224, 224, 3)
)

base_model.trainable = False  # Freeze base model

# Add custom classification head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
predictions = Dense(3, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=predictions)

# Compile with Adam optimizer
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Phase 1 training
history_phase1 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5,
    verbose=1
)

print("✅ Phase 1 training complete!")

# ==================== PHASE 2: FINE-TUNE TOP LAYERS ====================
print("\n🔓 Phase 2: Fine-tuning top layers of base model...")

# Unfreeze only the last 30 layers for fine-tuning
base_model.trainable = True
for layer in base_model.layers[:-30]:
    layer.trainable = False

# Recompile with very low learning rate
model.compile(
    optimizer=Adam(learning_rate=1e-5),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Phase 2 fine-tuning
history_phase2 = model.fit(
    train_data,
    validation_data=val_data,
    epochs=5,
    verbose=1
)

print("✅ Phase 2 fine-tuning complete!")

# ==================== COMBINE HISTORIES ====================
# Combine both training phases for visualization
combined_history = {
    'accuracy': history_phase1.history['accuracy'] + history_phase2.history['accuracy'],
    'val_accuracy': history_phase1.history['val_accuracy'] + history_phase2.history['val_accuracy'],
    'loss': history_phase1.history['loss'] + history_phase2.history['loss'],
    'val_loss': history_phase1.history['val_loss'] + history_phase2.history['val_loss']
}

# ==================== SAVE MODEL ====================
# Use SavedModel format (not deprecated .h5)
model_path = "plant_leaf_model"
model.save(model_path)
print(f"\n💾 Model saved to {model_path}/ (SavedModel format)")

# ==================== GENERATE TRAINING PLOTS ====================
print("\n📊 Generating training plots...")

# Create a figure with 2 subplots
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Plot 1: Accuracy
axes[0].plot(combined_history['accuracy'], label='Training Accuracy', linewidth=2, marker='o')
axes[0].plot(combined_history['val_accuracy'], label='Validation Accuracy', linewidth=2, marker='s')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Model Accuracy Over Time', fontsize=14, fontweight='bold')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_ylim([0, 1])

# Plot 2: Loss
axes[1].plot(combined_history['loss'], label='Training Loss', linewidth=2, marker='o')
axes[1].plot(combined_history['val_loss'], label='Validation Loss', linewidth=2, marker='s')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].set_title('Model Loss Over Time', fontsize=14, fontweight='bold')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()

# Create screenshots directory if it doesn't exist
os.makedirs("screenshots", exist_ok=True)

# Save the plot
plot_path = "screenshots/training_history.png"
plt.savefig(plot_path, dpi=150, bbox_inches='tight')
print(f"📈 Training plot saved to {plot_path}")

# Also display summary
print("\n" + "="*60)
print("🎉 TRAINING SUMMARY")
print("="*60)
print(f"Final Training Accuracy: {combined_history['accuracy'][-1]:.4f}")
print(f"Final Validation Accuracy: {combined_history['val_accuracy'][-1]:.4f}")
print(f"Final Training Loss: {combined_history['loss'][-1]:.4f}")
print(f"Final Validation Loss: {combined_history['val_loss'][-1]:.4f}")
print("="*60)

print("\n✅ All done! Your model is ready for deployment.")
