# 🌿 Plant Leaf Classifier

**An end-to-end AI-powered web application that identifies plant species from leaf images using deep learning.** The system provides confidence-based predictions for all classes, handles unknown inputs responsibly, and presents results through a clean, modern, and mobile-responsive web interface.

---

## ✨ Features

* **🎯 Species Recognition:** Classifies leaf images as Mango, Neem, or Tulsi with high accuracy
* **📊 Confidence Visualization:** Shows confidence bars for all three species (transparent model reasoning)
* **🛡️ Robust Error Handling:** Graceful error messages using Flask flash notifications
* **🌍 Unknown Samples:** Handles out-of-distribution inputs (non-leaf images) responsibly  
* **💊 Educational Insights:** Displays medicinal and culinary uses for detected plants
* **📱 Mobile Responsive:** Clean, modern UI that works on phones, tablets, and desktops
* **🔒 Secure Uploads:** Robust file handling with validation and size limits (5MB)
* **⚡ Fast Predictions:** Lightweight MobileNetV2 model for quick inference

---

## 🏆 Tech Stack

| Component | Technology |
| :--- | :--- |
| **ML/AI** | TensorFlow, Keras, MobileNetV2 (Transfer Learning) |
| **Backend** | Python, Flask |
| **Frontend** | HTML5, CSS3, Responsive Design |
| **Data Processing** | NumPy, Pillow |
| **Deployment** | Docker, Render/Railway ready |
| **Version Control** | Git & GitHub |

---

## 📁 Project Structure

```
plant-leaf-classifier/
├── web/
│   ├── app.py                      # Flask application (routes, predictions)
│   ├── templates/
│   │   └── app.html                # Single template (home + upload + results)
│   └── static/
│       └── uploads/                # User-uploaded images
│
├── src/
│   ├── train.py                    # Model training (with data augmentation & fine-tuning)
│   └── predict.py                  # Standalone prediction script for testing
│
├── dataset/
│   └── train/
│       ├── mango/                  # Mango leaf images
│       ├── neem/                   # Neem leaf images
│       └── tulsi/                  # Tulsi leaf images
│
├── screenshots/
│   └── training_history.png        # Training accuracy/loss plots
│
├── plant_leaf_model/               # Trained model (SavedModel format)
├── requirements.txt                # Python dependencies
├── Dockerfile                      # Docker container definition
├── Procfile                        # Deployment configuration
└── README.md                       # This file
```

---

## 🚀 Quick Start

### 1️⃣ Clone & Setup

```bash
git clone https://github.com/YOUR_USERNAME/plant-leaf-classifier.git
cd plant-leaf-classifier
```

### 2️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 3️⃣ Run the Web App

```bash
cd web
python app.py
```

Open your browser to `http://localhost:5000`

### 4️⃣ Train the Model (Optional)

If you want to retrain on your own dataset:

```bash
cd src
python train.py
```

This will:
- Load images from `../dataset/train/`
- Apply aggressive data augmentation (rotation, zoom, brightness, etc.)
- Train with frozen base model (5 epochs)
- Fine-tune top 30 layers with low learning rate (5 epochs)
- Save to SavedModel format (not deprecated .h5)
- Generate training plots to `../screenshots/training_history.png`

### 5️⃣ Test with Static Images

```bash
cd src
python predict.py
```

Place test images in `../test_images/` folder.

---

## 🧠 Model Architecture

The model uses **transfer learning** with a pre-trained MobileNetV2 backbone:

```
Input (224×224×3)
    ↓
MobileNetV2 (ImageNet weights, frozen)
    ↓
GlobalAveragePooling2D
    ↓
Dense(128, relu)
    ↓
Dense(3, softmax)  ← Outputs [mango, neem, tulsi] probabilities
```

**Training Strategy (Two-Phase):**

1. **Phase 1:** Freeze base model, train custom head (5 epochs, lr=0.001)
2. **Phase 2:** Unfreeze last 30 layers, fine-tune with low lr (5 epochs, lr=1e-5)

This approach prevents overfitting on small datasets and maintains strong pre-trained features.

---

## 🎨 Data Augmentation

Applied to training data only (not validation):

- **Rotation:** ±20°
- **Horizontal Flip:** 50%
- **Zoom:** 0.8–1.2×
- **Brightness:** ±20%
- **Shear:** 0.1

These augmentations dramatically reduce overfitting and improve model robustness to real-world variations.

---

## 📊 Results

After training with two-phase fine-tuning:
- Training accuracy typically reaches **92–97%**
- Validation accuracy typically reaches **85–92%**
- Confidence threshold set to **70%** to flag uncertain predictions

See `screenshots/training_history.png` for detailed accuracy/loss curves.

---

## 🌐 Deployment

### Option 1: Docker (Local Testing)

```bash
docker build -t plant-leaf-classifier .
docker run -p 5000:5000 plant-leaf-classifier
```

### Option 2: Render.com (Free)

1. Fork this repo to your GitHub account
2. Create a new **Web Service** on [Render](https://render.com)
3. Connect your GitHub repo
4. Set runtime to `Python 3.10`
5. Deploy! Render will auto-detect `Procfile`

### Option 3: Railway.app (Free)

1. Push to GitHub
2. Go to [Railway](https://railway.app) and link your repo
3. Railway auto-detects `Procfile` and deploys

### Environment Variables (for production)

Set these in your deployment platform dashboard:

```bash
FLASK_ENV=production
SECRET_KEY=your-secret-key-here
```

---

## 🔧 API Usage

### Prediction Endpoint

**POST** `/predict`

**Request:**
```html
<form method="POST" enctype="multipart/form-data">
    <input type="file" name="image" accept="image/*" required>
    <button type="submit">Predict</button>
</form>
```

**Response (HTML page):**
- Displays prediction result
- Shows confidence bars for all three classes
- Lists medicinal uses (if confident prediction)
- Displays uploaded image

**Error Handling:**
- Invalid file type → Flash message: "❌ Invalid file type..."
- No file selected → Flash message: "❌ Please select a file..."
- Non-leaf image → Prediction: "Unknown or unsupported leaf"

---

## 📈 Confidence Bar Visualization

The UI now shows confidence for **all three classes**, not just the top prediction:

```
📊 Model Confidence Scores
Mango    ████████░░░░░░░░░░░  75.3%
Neem     ████░░░░░░░░░░░░░░░  20.1%
Tulsi    ██░░░░░░░░░░░░░░░░░   4.6%
```

This makes the model's reasoning transparent and builds user trust.

---

## 🔐 Security Features

- ✅ File type validation (PNG, JPG, JPEG only)
- ✅ File size limit (5MB max)
- ✅ Secure filename handling (UUID)
- ✅ Uploads stored in isolated folder
- ✅ CSRF protection ready (can enable with Flask-WTF)

---

## 🎓 Learning Outcomes

This project demonstrates:

- **Deep Learning:** Transfer learning, fine-tuning, data augmentation
- **Web Development:** Flask, Jinja2 templates, form handling
- **MLOps:** Model versioning (SavedModel), training visualization, requirements management
- **DevOps:** Docker containerization, cloud deployment (Render/Railway)
- **UI/UX:** Responsive design, accessibility, error handling best practices
- **Python Best Practices:** Modular code, error handling, secure file operations

---

## 📝 Future Enhancements

- [ ] **Expand to more species** (38+ classes from PlantVillage dataset)
- [ ] **Disease detection** (separate model or multi-task learning)
- [ ] **Prediction history** (SQLite database + history route)
- [ ] **REST API** (add `/api/predict` endpoint for mobile apps)
- [ ] **Batch processing** (upload multiple images at once)
- [ ] **Model explainability** (LIME/SHAP for explaining predictions)
- [ ] **User authentication** (login system for curating predictions)
- [ ] **CI/CD pipeline** (GitHub Actions for automated testing)

---

## 🐛 Troubleshooting

**Model file not found error:**
```
FileNotFoundError: plant_leaf_model not found
```
Solution: Run `python src/train.py` to train the model first.

**Module not found (tensorflow, flask, etc.):**
```bash
pip install -r requirements.txt
```

**Port 5000 already in use:**
```bash
python web/app.py --port 5001
```

---

## 📄 License

This project is open source and available under the **MIT License**.

---

## 🙏 Credits

- Dataset inspiration: [PlantVillage](https://plantvillage.psu.edu/)
- Model: [MobileNetV2](https://arxiv.org/abs/1801.04381) (Google)
- Framework: [TensorFlow](https://tensorflow.org) & [Flask](https://flask.palletsprojects.com)

---

## ✉️ Questions?

Feel free to open an **Issue** on GitHub!

**Happy leaf classifying! 🍃**  

