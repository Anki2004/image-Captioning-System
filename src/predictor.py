import numpy as np
import pickle
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.models import Model
from src.model import build_model

# Load the tokenizer
try:
    with open('models/tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
except (EOFError, FileNotFoundError):
    print("Error: tokenizer.pickle is missing or corrupted. Please retrain the model.")
    exit(1)

# Load max_length
try:
    with open('models/max_length.txt', 'r') as f:
        max_length = int(f.read().strip())
except (ValueError, FileNotFoundError):
    print("Error: max_length.txt is missing or invalid. Please retrain the model.")
    exit(1)

# Load InceptionV3 model for feature extraction
inception_model = InceptionV3(weights='imagenet')
inception_model = Model(inception_model.input, inception_model.layers[-2].output)

# Rebuild the model
vocab_size = len(tokenizer.word_index) + 1
model = build_model(vocab_size, max_length)

# Load the weights
try:
    model.load_weights('models/model.h5')
except:
    print("Error: Failed to load model weights. Please retrain the model.")
    exit(1)

def extract_features(image):
    features = inception_model.predict(image)
    return features

def word_for_id(integer, tokenizer):
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None

def predict_caption(image):
    # Extract features from the image
    features = extract_features(image)
    
    # Generate caption
    in_text = 'startseq'

    for _ in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        
        yhat = model.predict([features, sequence], verbose=0)
        yhat = np.argmax(yhat)
        
        word = word_for_id(yhat, tokenizer)
        
        if word is None or word == 'endseq':
            break
            
        in_text += ' ' + word

    final_caption = in_text.replace('startseq', '').strip()
    
    return final_caption