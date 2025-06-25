from flask import Flask, render_template, request, redirect, url_for, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras import models, layers
from tensorflow.keras.datasets import mnist
import numpy as np
import os, base64, io
from PIL import Image

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs('models', exist_ok=True)

MODELS = {
    "model1": "models/model1.h5",
    "model2": "models/model2.h5"
}

@app.route("/")
def index():
    return render_template("index.html", models=MODELS)

@app.route("/model/<model_id>")
def model_draw(model_id):
    if model_id not in MODELS:
        return "Model not found", 404
    return render_template("model_draw.html", model_id=model_id, models=MODELS)

@app.route("/add_model")
def add_model():
    return render_template("model_builder.html", models=MODELS)

@app.route("/save_model", methods=["POST"])
def save_model():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train = x_train / 255.0
    x_test = x_test / 255.0

    model = models.Sequential()
    first_layer = True
    uses_conv = False

    i = 0
    while True:
        layer_type = request.form.get(f"layer_type_{i}")
        if not layer_type:
            break

        units = request.form.get(f"units_{i}")
        activation = request.form.get(f"activation_{i}") or None

        if layer_type == "dense":
            if first_layer:
                model.add(layers.Flatten(input_shape=(28, 28)))
                first_layer = False
            model.add(layers.Dense(int(units), activation=activation))

        elif layer_type == "conv":
            uses_conv = True
            filters = int(units)
            if first_layer:
                model.add(layers.Conv2D(filters, (3, 3), activation=activation, input_shape=(28, 28, 1)))
                first_layer = False
            else:
                model.add(layers.Conv2D(filters, (3, 3), activation=activation))

        elif layer_type == "maxpool":
            model.add(layers.MaxPooling2D((2, 2)))

        elif layer_type == "flatten":
            model.add(layers.Flatten())

        i += 1
    model.add(layers.Dense(10, activation='softmax'))

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    if uses_conv:
        x_train = x_train.reshape(-1, 28, 28, 1)
        x_test = x_test.reshape(-1, 28, 28, 1)

    model.fit(x_train, y_train, epochs=5, validation_data=(x_test, y_test))

    model_id = f"model{len(MODELS) + 1}"
    path = f"models/{model_id}.h5"
    model.save(path)

    MODELS[model_id] = path  

    return redirect(url_for("model_draw", model_id=model_id))

@app.route('/predict_canvas/<model_id>', methods=['POST'])
def predict_canvas(model_id):
    if model_id not in MODELS:
        return jsonify({'error': 'Model not found'}), 404

    data = request.get_json()
    if 'image' not in data:
        return jsonify({'error': 'No image found in request'}), 400

    image_data = data['image'].split(',')[1]
    image_bytes = base64.b64decode(image_data)
    image = Image.open(io.BytesIO(image_bytes)).convert('L')  
    image = image.resize((28, 28))

    image_np = np.array(image)
    image_np = 255 - image_np  
    image_np = image_np / 255.0 
    image_np = image_np.reshape(1, 28, 28, 1) 

    model = load_model(MODELS[model_id])
    prediction = model.predict(image_np)
    predicted_class = int(np.argmax(prediction))

    return jsonify({'prediction': predicted_class})


app.run(debug=True)