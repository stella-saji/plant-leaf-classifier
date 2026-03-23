# 🌿 Plant Leaf Classifier  

**An end-to-end AI-powered web application that identifies plant species from leaf images using deep learning.** The system provides confidence-based predictions, handles unknown inputs responsibly, and presents results through a clean, modern web interface.  

---

## Features  

* **Species Recognition:** Classifies leaf images as Neem, Mango, or Tulsi.  
* **Confidence Scores:** Displays exactly how sure the AI is about its choice.  
* **Out-of-Distribution Handling:** Detects unknown or unsupported images (e.g., animals, objects).  
* **Educational Insights:** Shows medicinal and common uses of detected plants.  
* **Interactive UI:** A sleek Flask-based web interface with a loading spinner.  
* **Dark Mode:** Easy on the eyes for focused work.  
* **Secure Uploads:** Robust handling for user-submitted images.  

---

## Tech Stack  

| Layer | Technologies |  
| :--- | :--- |  
| **AI / ML** | TensorFlow, Keras, MobileNetV2 (Transfer Learning) |  
| **Backend** | Python, Flask |  
| **Frontend** | HTML, CSS, JavaScript |  
| **Utilities** | NumPy, OpenCV |  
| **Version Control** | Git & GitHub |  

---

## Project Structure  

```plaintext
plant-leaf-classifier/
│
├── web/
│   ├── app.py
│   ├── templates/
│   │   ├── home.html
│   │   └── predict.html
│   └── static/
│       └── uploads/
│
├── dataset/
├── src/
│   ├── train.py
│   └── predict.py
│
├── screenshots/
│   ├── home.png
│   └── result.png
│
├── README.md
└── .gitignore
```

---

## Getting Started  

**1. Clone the repository** 

git clone https://github.com/stella-saji/plant-leaf-classifier.git

cd plant-leaf-classifier



**2. Set up a Virtual Environment** 

python -m venv venv

On Windows:

venv\Scripts\activate

On Mac/Linux:

source venv/bin/activate


**3. Install Dependencies**

pip install tensorflow flask numpy opencv-python


**4. Launch the App** 

python web/app.py

*Visit [http://127.0.0.1:5000](http://127.0.0.1:5000) in your browser.* ---

## How the Model Works  

* **Architecture:**

  Uses Transfer Learning with a pre-trained MobileNetV2 for high accuracy with lower  computational cost.

* **Logic:**

   Outputs class probabilities using a Softmax activation function.  

* **Responsibility:**

  Applies a confidence threshold to reject low-confidence or "out-of-distribution" inputs, ensuring the AI doesn't provide false identifications.  

---

## Future Enhancements  

* Mobile-friendly UI optimization.
  
* Support for a wider variety of plant species.
   
* Identifying leaf diseases (sick vs. healthy).
  
* Cloud deployment on platforms like AWS or Azure.  

---

## Author  

**Stella Saji**

*Aspiring AI Engineer & Developer* Built with curiosity, persistence, and a focus on efficient learning.

---

## Acknowledgements  

* TensorFlow & Keras community
  
* Flask documentation
   
* Open-source plant image datasets  

