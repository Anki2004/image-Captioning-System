import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pickle
from src.data_loader import load_data
from src.model import build_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping

def train_model(captions_file, images_dir, epochs=20, batch_size=32):
    # Load and prepare data
    train_data, val_data, tokenizer, max_length, vocab_size = load_data(captions_file, images_dir)

    # Build model
    model = build_model(vocab_size, max_length)

    # Define callbacks
    checkpoint = ModelCheckpoint('models/model.h5', save_best_only=True, monitor='val_loss', mode='min')
    early_stopping = EarlyStopping(patience=5, monitor='val_loss', mode='min')

    # Train model
    history = model.fit(
        train_data[0], train_data[1],
        epochs=epochs,
        batch_size=batch_size,
        validation_data=val_data,
        callbacks=[checkpoint, early_stopping]
    )

    # Save tokenizer
    with open('models/tokenizer.pickle', 'wb') as handle:
        pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Save max_length
    with open('models/max_length.txt', 'w') as f:
        f.write(str(max_length))

    return history

if __name__ == "__main__":
    train_model('data\caption\captions.txt', 'data/images')