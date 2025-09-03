import os
import numpy as np
import librosa

def extract_features(file_path):
    """Extract comprehensive features from audio file using librosa"""
    try:
        # Load audio file
        audio, sample_rate = librosa.load(file_path, sr=None)
        
        # Extract multiple audio features
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        chroma = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        mel = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        contrast = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        tonnetz = librosa.feature.tonnetz(y=librosa.effects.harmonic(audio), sr=sample_rate)
        
        # Calculate statistics for each feature
        features = []
        
        # MFCC statistics
        features.extend(np.mean(mfccs.T, axis=0))
        features.extend(np.std(mfccs.T, axis=0))
        
        # Chroma statistics
        features.extend(np.mean(chroma.T, axis=0))
        features.extend(np.std(chroma.T, axis=0))
        
        # Mel spectrogram statistics
        features.extend(np.mean(mel.T, axis=0))
        features.extend(np.std(mel.T, axis=0))
        
        # Spectral contrast statistics
        features.extend(np.mean(contrast.T, axis=0))
        features.extend(np.std(contrast.T, axis=0))
        
        # Tonnetz statistics
        features.extend(np.mean(tonnetz.T, axis=0))
        features.extend(np.std(tonnetz.T, axis=0))
        
        # Additional features
        features.append(librosa.feature.rms(y=audio).mean())
        features.append(librosa.feature.zero_crossing_rate(audio).mean())
        features.append(librosa.feature.spectral_centroid(y=audio, sr=sample_rate).mean())
        features.append(librosa.feature.spectral_bandwidth(y=audio, sr=sample_rate).mean())
        features.append(librosa.feature.spectral_rolloff(y=audio, sr=sample_rate).mean())
        
        return np.array(features)
        
    except Exception as e:
        print(f"Error extracting features from {file_path}: {e}")
        return None

def load_ravdess_data(dataset_path):
    """Load RAVDESS dataset and extract features and labels with proper parsing"""
    features = []
    labels = []
    
    print("Scanning dataset directory...")
    
    for actor_dir in os.listdir(dataset_path):
        actor_path = os.path.join(dataset_path, actor_dir)
        if os.path.isdir(actor_path):
            print(f"Processing actor: {actor_dir}")
            
            for file in os.listdir(actor_path):
                if file.endswith('.wav'):
                    try:
                        # Parse filename according to RAVDESS format
                        parts = file[:-4].split('-')  # Remove .wav and split by hyphen
                        
                        if len(parts) >= 4:
                            modality = parts[0]  # 03 = audio-only
                            vocal_channel = parts[1]  # 01 = speech, 02 = song
                            emotion = int(parts[2])  # Emotion code (1-8)
                            intensity = int(parts[3])  # Intensity code (1-2)
                            
                            # Only process speech audio files
                            if vocal_channel == '01' and modality == '03':
                                file_path = os.path.join(actor_path, file)
                                extracted_features = extract_features(file_path)
                                
                                if extracted_features is not None:
                                    features.append(extracted_features)
                                    labels.append((emotion, intensity))
                    
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing filename {file}: {e}")
                        continue
    
    print(f"Successfully loaded {len(features)} samples")
    return np.array(features), np.array(labels)
