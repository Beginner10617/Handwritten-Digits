import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
import sys

def predict_digit(image_path, model):
    model = load_model(model)
    image = Image.open(image_path).convert('L')  # grayscale
    image = image.resize((28, 28))
    image = np.array(image)
    image = 255 - image  # invert if background is black
    image = image / 255.0
    image = image.reshape(1, 28, 28, 1)
    prediction = model.predict(image)
    return np.argmax(prediction)

if __name__ == "__main__":
    print(predict_digit(sys.argv[1], sys.argv[2]))
