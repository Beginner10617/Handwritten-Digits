import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model

model = load_model(sys.argv[1])  # load model passed via CLI
image_number = 1

while os.path.isfile(f"test/{image_number}.webp"):
    try:
        # Load and preprocess the image
        img = cv2.imread(f"test/{image_number}.webp", cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Image could not be loaded.")
        
        img = cv2.resize(img, (28, 28))  # resize to match model input
        img = 255 - img  # invert image so white background, black digit
        img = img / 255.0  # normalize to [0, 1]
        img = img.reshape(1, 28, 28, 1)  # reshape to (1, 28, 28, 1)

        # Predict
        prediction = model.predict(img)
        print(f"[{image_number}] This digit is probably a {np.argmax(prediction)}")

        # Show image
        plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
        plt.title(f"Predicted: {np.argmax(prediction)}")
        plt.show()

    except Exception as e:
        print(f"Error with image {image_number}: {e}")
    
    finally:
        image_number += 1
