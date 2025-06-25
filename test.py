import os
import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
model = load_model(sys.argv[1])  
image_number = 1
while os.path.isfile(f"test/{image_number}.webp"):
    try:
        img = cv2.imread(f"test/{image_number}.webp", cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise ValueError("Image could not be loaded.")
        img = cv2.resize(img, (28, 28))  
        img = 255 - img  
        img = img / 255.0 
        img = img.reshape(1, 28, 28, 1)  
        prediction = model.predict(img)
        print(f"[{image_number}] This digit is probably a {np.argmax(prediction)}")
        plt.imshow(img.reshape(28, 28), cmap=plt.cm.binary)
        plt.title(f"Predicted: {np.argmax(prediction)}")
        plt.show()

    except Exception as e:
        print(f"Error with image {image_number}: {e}")
    
    finally:
        image_number += 1
