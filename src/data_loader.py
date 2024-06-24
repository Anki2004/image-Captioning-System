# import numpy as np
# from tensorflow.keras.preprocessing.text import Tokenizer
# from tensorflow.keras.preprocessing.sequence import pad_sequences
# from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
# from tensorflow.keras.preprocessing.image import load_img, img_to_array
# from tensorflow.keras.models import Model


# def load_data(captions_file, image_dir):
#     with open(captions_file, 'r') as f:
#         captions = f.read().split('\n')

#     img_to_captions = {}
#     for caption in captions:
#         if caption.strip():  # Skip empty lines
#             parts = caption.split(',')
#             if len(parts) >= 2:
#                 img = parts[0].strip()
#                 cap = ','.join(parts[1:]).strip()  # Join all parts after the first comma
#                 if img not in img_to_captions:
#                     img_to_captions[img] = []
#                 img_to_captions[img].append(cap)
#             else:
#                 print(f"Skipping invalid line: {caption}")

#     tokenizer = Tokenizer()
#     tokenizer.fit_on_texts([cap for caps in img_to_captions.values() for cap in caps])

#     inception = InceptionV3(weights = 'imagenet')
#     inception_model = Model(inception.input, inception.layers[-2].output)

#     img_features = {}
#     for img in img_to_captions.keys():
#         img_path = f'{image_dir}/{img}'
#         img = load_img(img_path, target_size = (299, 299))
#         img = img_to_array(img)
#         img = np.expand_dims(img, axis = 0)
#         img = preprocess_input(img)
#         features = inception_model.predict(img)
#         img_features[img] = features
    
#     X1, X2, y = [], [], []
#     for img, caps in img_to_captions.items():
#         for cap in caps:
#             seq = tokenizer.texts_to_sequences([cap])[0]
#             for i in range(1, len(seq)):
#                 in_seq, out_seq= seq[:i], seq[i]
#                 in_seq = pad_sequences([in_seq], maxlen = 34)[0]
#                 out_seq = to_categorical([out_seq], num_classes = len(tokenizer.word_index)+1)[0]
#                 X1.append(img_features[img][0])
#                 X2.append(in_seq)
#                 y.append(out_seq)

#     X1, X2, y = np.array(X1), np.array(X2), np.array(y)

#     split = int(0.8 * len(X1))
#     train_data = ([X1[:split], X2[:split]], y[:split])
#     val_data = ([X1[split:], X2[split:]], y[split:])\
    
#     return train_data, val_data, tokenizer


import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.models import Model
from PIL import Image
import os

def load_data(captions_file, images_dir):
    # Load captions
    with open(captions_file, 'r') as f:
        captions = f.read().split('\n')

    # Process captions
    img_to_captions = {}
    for caption in captions:
        if caption.strip():  # Skip empty lines
            parts = caption.split(',')
            if len(parts) >= 2:
                img = parts[0].strip()
                cap = ','.join(parts[1:]).strip()  # Join all parts after the first comma
                if img not in img_to_captions:
                    img_to_captions[img] = []
                img_to_captions[img].append(cap)
            else:
                print(f"Skipping invalid line: {caption}")

    # Tokenize captions
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts([cap for caps in img_to_captions.values() for cap in caps])

    # Load images and extract features
    inception = InceptionV3(weights='imagenet')
    inception_model = Model(inception.input, inception.layers[-2].output)

    img_features = {}
    for img in img_to_captions.keys():
        img_path = os.path.join(images_dir, img)
        if os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB')
            image = image.resize((299, 299))
            image = np.array(image)
            image = np.expand_dims(image, axis=0)
            image = preprocess_input(image)
            features = inception_model.predict(image)
            img_features[img] = features
        else:
            print(f"Image not found: {img_path}")

    # Prepare training data
    max_length = max(len(cap.split()) for caps in img_to_captions.values() for cap in caps)
    vocab_size = len(tokenizer.word_index) + 1

    X1, X2, y = [], [], []
    for img, caps in img_to_captions.items():
        if img in img_features:
            for cap in caps:
                seq = tokenizer.texts_to_sequences([cap])[0]
                for i in range(1, len(seq)):
                    in_seq, out_seq = seq[:i], seq[i]
                    in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
                    out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]
                    X1.append(img_features[img][0])
                    X2.append(in_seq)
                    y.append(out_seq)

    X1, X2, y = np.array(X1), np.array(X2), np.array(y)

    # Split into train and validation sets
    split = int(0.8 * len(X1))
    train_data = ([X1[:split], X2[:split]], y[:split])
    val_data = ([X1[split:], X2[split:]], y[split:])

    return train_data, val_data, tokenizer, max_length, vocab_size