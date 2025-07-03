# 🧠 Handwritten Digit Recognition Web App

An interactive web application to **create, train, test, and use neural networks** for handwritten digit classification using the [MNIST dataset](http://yann.lecun.com/exdb/mnist/). Built with **Flask** and **TensorFlow/Keras**, this app allows you to:

- ✏️ Draw digits on a canvas and recognize them
- 📤 Upload digit images for classification
- 🏗️ Visually build and train your own deep learning models
- 📊 View model performance (accuracy & loss)

---

## 🔍 Dataset: MNIST

- 70,000 grayscale images of handwritten digits
- Each image is 28×28 pixels
- Split into:
  - 60,000 training samples
  - 10,000 test samples

---

## 📦 Features

### ✅ Pre-trained Models
- **Model 1**: Fully connected network (MLP)
  - `Flatten → Dense(128) → Dense(128) → Dense(10)`
  - Accuracy: **97.69%**, Loss: **0.0780**
- **Model 2**: Convolutional neural network (CNN)
  - `Conv2D(32) → MaxPool → Conv2D(64) → MaxPool → Flatten → Dense(128) → Dense(10)`
  - Accuracy: **98.95%**, Loss: **0.0354**

### ✅ Live Digit Prediction
- Use mouse to draw a digit on a canvas
- Or upload an image of a digit
- Instant prediction using any loaded model

### ✅ Custom Model Builder
- Add layers: `Dense`, `Conv2D`, `MaxPooling`, `Flatten`
- Specify layer parameters (units, activation)
- Train on MNIST and view predictions
- Saved models appear in the sidebar menu

---

## 🚀 Getting Started

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/handwriting-recognition-flask.git
cd handwriting-recognition-flask
```

### 2. Set Up Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate   # on Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run the App
```bash
python app.py
```

### 5. Open in browser
```bash
http://127.0.0.1:5000/
```

---
## 🧠 Future Enhancements

- Save training graphs (accuracy/loss over epochs)
- User login & personal model history
- Delete/edit existing models
- Train on custom datasets (e.g., EMNIST)

---

## 📬 Contact

- **Email:** wasihusain23@iitk.ac.in
- **GitHub**: [https://github.com/Beginner10617](https://github.com/Beginner10617)
