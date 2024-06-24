import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from flask import Flask, request, jsonify
from src.predictor import predict_caption
from utils.preprocessing import preprocess_image

import io
from PIL import Image
app = Flask(__name__)

@app.route('/predict', methods = ['POST'])

def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    image_file = request.files['image']
    image = Image.open(io.BytesIO(image_file.read()))
    processed_image = preprocess_image(image)

    caption = predict_caption(processed_image)

    return jsonify({'caption':caption})

if __name__ == '__main__':
    app.run(debug=True)