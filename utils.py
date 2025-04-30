import os
import numpy as np
import random
import tensorflow as tf
import keras
from tcn import TCN
from tensorflow.keras.models import model_from_json
import matplotlib.pyplot as plt
import librosa
import librosa.display
import mirdata
from pathlib import Path
from sklearn.model_selection import train_test_split
from tqdm import tqdm


# Define label mapping for 11 instrument classes
instrument_labels = {
    'cel': 0,  # cello
    'cla': 1,  # clarinet
    'flu': 2,  # flute
    'gac': 3,  # acoustic guitar
    'gel': 4,  # electric guitar
    'org': 5,  # organ
    'pia': 6,  # piano
    'sax': 7,  # saxophone
    'tru': 8,  # trumpet
    'vio': 9,  #violin
    'voi': 10, # vocal
}



# Extract embeddings from an audio waveform using YAMNet
def extract_yamnet_embedding(wav_data, yamnet):
    
    scores, embeddings, spectrogram = yamnet(wav_data)

    return embeddings  





# Extract all embeddings and labels from a folder of .wav files, organized by instrument type
def extract_embeddings_from_folder(root_folder, yamnet):
    all_embeddings = []
    all_labels = []
    
    for label_name in sorted(os.listdir(root_folder)):
        label_path = os.path.join(root_folder, label_name)
        if not os.path.isdir(label_path):
            continue
        
        print(f"Processing {label_name}...")
        label_index = instrument_labels[label_name]

        for filename in os.listdir(label_path):
            if not filename.endswith('.wav'):
                continue

            file_path = os.path.join(label_path, filename)

            waveform, sr = librosa.load(file_path, sr=16000)  # Load waveform (resampled to 16kHz)

            embeddings = extract_yamnet_embedding(waveform, yamnet)
            embeddings = embeddings.numpy()

            for embed in embeddings:  # Store frame-wise embeddings with corresponding label
                all_embeddings.append(embed)
                all_labels.append(label_index)
    
    return np.array(all_embeddings), np.array(all_labels)






# Convert frame-level embeddings into overlapping sequences for TCN training
def create_sequences(X, y, sequence_length):
    sequences = []
    labels = []

    for i in range(len(X) - sequence_length + 1):
        seq = X[i:i+sequence_length]
        label = y[i+sequence_length-1]  # Assign label of the last frame
        sequences.append(seq)
        labels.append(label)

    return np.array(sequences), np.array(labels)


