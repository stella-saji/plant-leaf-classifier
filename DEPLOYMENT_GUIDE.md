# 🚀 Deployment Guide

This guide explains how to deploy your Plant Leaf Classifier on popular free platforms.

---

## Option 1: Render.com (Recommended - Free)

Render offers a generous free tier and auto-deployment from GitHub.

### Steps:

1. **Push your code to GitHub** (if not already done)
   ```bash
   git add .
   git commit -m "Add plant leaf classifier"
   git push origin main
   ```

2. **Create a Render account**
   - Go to [render.com](https://render.com)
   - Sign up with GitHub

3. **Create a new Web Service**
   - Click "New +" → "Web Service"
   - Connect your GitHub repository
   - Select the repo containing this project

4. **Configure the service**
   - **Name:** `plant-leaf-classifier`
   - **Environment:** `Python 3.10`
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `python web/app.py`
   - **Free Plan:** Select it

5. **Set Environment Variables**
   - In Render dashboard: Settings → Environment
   - Add:
     ```
     FLASK_ENV=production
     SECRET_KEY=your-secret-key-change-this
     ```

6. **Deploy!**
   - Click "Create Web Service"
   - Render builds and deploys automatically
   - Your app will be live at `https://your-service-name.onrender.com`

### Important Notes:
- Free tier spins down after 15 minutes of inactivity (first request takes ~30 seconds)
- Free tier has limited compute (512MB RAM)
- Uploads are stored on ephemeral storage (deleted after restart)
- For persistent storage, upgrade or use cloud storage (AWS S3, etc.)

---

## Option 2: Railway.app (Also Free)

Railway is another great option with similar features to Render.

### Steps:

1. **Push to GitHub** (same as above)

2. **Create Railway account**
   - Go to [railway.app](https://railway.app)
   - Sign up with GitHub

3. **Create new project**
   - Click "New Project" → "Deploy from GitHub repo"
   - Select your repository
   - Railway auto-detects Python and `Procfile`

4. **Configure environment**
   - Project Settings → Variables
   - Add:
     ```
     FLASK_ENV=production
     SECRET_KEY=your-secret-key-change-this
     PORT=5000
     ```

5. **Deploy**
   - Railway automatically deploys on every push
   - View logs: Railway Dashboard → Deployments tab

### Railway Free Tier:
- $5 monthly free credit (usually covers ~50-100 requests/month)
- After free credit, you pay as you go

---

## Option 3: Docker + Local VPS

If you want more control, use Docker on a cheap VPS (DigitalOcean, Linode, etc.)

### Build Docker image:

```bash
docker build -t plant-leaf-classifier .
```

### Run locally (for testing):

```bash
docker run -p 5000:5000 plant-leaf-classifier
```

### Deploy to VPS:

1. Create a VPS (DigitalOcean droplet, ~$4–5/month)
2. Install Docker on the VPS
3. Push image to Docker Hub:
   ```bash
   docker tag plant-leaf-classifier:latest YOUR_USERNAME/plant-leaf-classifier:latest
   docker push YOUR_USERNAME/plant-leaf-classifier:latest
   ```
4. SSH into VPS and run:
   ```bash
   docker pull YOUR_USERNAME/plant-leaf-classifier:latest
   docker run -d -p 80:5000 YOUR_USERNAME/plant-leaf-classifier:latest
   ```

Your app is now live on your VPS's IP!

---

## Option 4: Hugging Face Spaces (Simplest - Gradio)

If you want minimal setup, use Hugging Face Spaces with Gradio.

### Create `app_gradio.py`:

```python
import gradio as gr
import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing import image
from PIL import Image
import io

# Load model
model = tf.keras.models.load_model("plant_leaf_model")
class_names = ["Mango", "Neem", "Tulsi"]

plant_info = {
    "Mango": "Rich in vitamins A and C. Improves digestion.",
    "Neem": "Antibacterial and antifungal. Used in skincare.",
    "Tulsi": "Relieves cold and cough. Reduces stress."
}

def predict(img):
    if img is None:
        return "No image provided"
    
    # Convert PIL to array
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Predict
    preds = model.predict(img_array)[0]
    pred_class = class_names[np.argmax(preds)]
    confidence = np.max(preds)
    
    result = f"**{pred_class}** ({confidence*100:.1f}%)\n\n{plant_info[pred_class]}"
    return result

demo = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs="markdown",
    title="🌿 Plant Leaf Classifier"
)

demo.launch()
```

### Deploy:

1. Push code to GitHub (including `app_gradio.py`)
2. Go to [huggingface.co/spaces](https://huggingface.co/spaces)
3. Click "Create new Space" → "Gradio"
4. Connect your GitHub repo
5. It auto-deploys!

---

## Environment Variables Cheat Sheet

| Variable | Purpose | Example |
| --- | --- | --- |
| `FLASK_ENV` | Development or production | `production` |
| `SECRET_KEY` | For session management | `random-secret-string` |
| `PORT` | Port to run on | `5000` |
| `DEBUG` | Enable debug mode (NOT for production!) | `False` |

---

## Troubleshooting Deployment

**"ModuleNotFoundError: No module named 'tensorflow'"**
- Make sure `requirements.txt` is in the root directory
- Render/Railway auto-detect and install it

**"Port already in use"**
- Use dynamic port: `os.environ.get('PORT', 5000)`
- Update `app.py`:
  ```python
  if __name__ == "__main__":
      port = int(os.environ.get('PORT', 5000))
      app.run(host='0.0.0.0', port=port)
  ```

**"Model file not found"**
- Ensure `plant_leaf_model/` is committed to GitHub (if not too large)
- Or train model after deployment

**"Out of memory"**
- Upgrade to paid tier
- Or use quantized model (TensorFlow Lite)

---

## Performance Tips

1. **Compress model:** Use TensorFlow Lite quantization
2. **Cache predictions:** Use Redis for repeated requests
3. **Background jobs:** Use Celery for long-running tasks
4. **CDN:** Use Cloudflare for faster static file delivery
5. **Monitor:** Set up error tracking with Sentry

---

## Next Steps

After deployment:

✅ Share your app link on LinkedIn/GitHub
✅ Add to portfolio website
✅ Get user feedback
✅ Continuously improve based on usage

---

**Happy deploying! 🚀**
