# 🌿 Plant Leaf Classifier

An end-to-end **AI-powered web application** that identifies plant species from leaf images using deep learning.  
The system provides **confidence-based predictions**, handles **unknown inputs responsibly**, and presents results through a **clean, modern web interface**.

**Built by Stella Saji** 💚

---

## ✨ Features

- 🍃 Classifies leaf images as **Neem**, **Mango**, or **Tulsi**
- 📊 Displays prediction **confidence score**
- ❓ Detects **unknown or unsupported images** (e.g., animals, objects)
- 💊 Shows **medicinal and common uses** of detected plants
- 🌐 Interactive **Flask web interface**
- 🎞️ Loading spinner during prediction
- 🌙 Dark mode support
- 🔐 Secure image upload handling

---

## 🧠 Tech Stack

| Layer | Technologies |
|------|-------------|
| **AI / ML** | TensorFlow, Keras, MobileNetV2 (Transfer Learning) |
| **Backend** | Python, Flask |
| **Frontend** | HTML, CSS, JavaScript |
| **Utilities** | NumPy, OpenCV |
| **Version Control** | Git & GitHub |

---

## 🏗️ Project Structure

plant-leaf-classifier/
│
├── web/
│ ├── app.py
│ ├── templates/
│ │ ├── home.html
│ │ └── predict.html
│ └── static/
│ └── uploads/
│
├── dataset/
├── src/
│ ├── train.py
│ └── predict.py
│
├── screenshots/
│ ├── home.png
│ └── result.png
│
├── README.md
└── .gitignore


---

## 🚀 How to Run Locally

### 1️⃣ Clone the repository
```bash
git clone https://github.com/stella-saji/plant-leaf-classifier.git
cd plant-leaf-classifier
python -m venv venv
venv\Scripts\activate
pip install tensorflow flask numpy opencv-python
python web/app.py
http://127.0.0.1:5000
```
---

## 🧪 How the Model Works

- Uses Transfer Learning with a pre-trained MobileNetV2
- Trained on labeled plant leaf images
- Outputs class probabilities using Softmax
- Applies a confidence threshold to avoid false predictions
- Rejects low-confidence and out-of-distribution inputs
 This ensures responsible and transparent AI behavior.

---

## 🌱 Future Enhancements 

- 📱 Mobile-friendly UI
- 🌿 Support for more plant species
- 🦠 Leaf disease detection
- ☁️ Cloud deployment
- 📊 Model performance visualization

---

## 👩‍💻 Author

Stella Saji
Aspiring AI Engineer & Developer
Built with curiosity, persistence, and a love for learning 🌿

---

## ⭐ Acknowledgements

- TensorFlow & Keras community
- Flask documentation
- Open-source plant image datasets