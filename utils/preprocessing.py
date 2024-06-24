from tensorflow.keras.application.inception_v3 import preprocess_input
from tensorflow.keras.preprocessing.image import img_to_array

import numpy as np

def preprocess_image(image):
    image = image.resize((299, 299))
    image = img_to_array(image)
    image = np.expand_dims(image, axis = 0)
    image = preprocess_input(image)
    return image