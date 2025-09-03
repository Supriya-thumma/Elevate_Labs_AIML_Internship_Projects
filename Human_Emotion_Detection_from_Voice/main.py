import os
import numpy as np
import pandas as pd
import librosa
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import LabelEncoder
import warnings
warnings.filterwarnings('ignore')

# Import utility functions
from utils import extract_features, load_ravdess_data

def train_emotion_model():
    """Main function to train the emotion detection model"""
    
    print("Loading RAVDESS dataset...")
    
    # Path to your RAVDESS dataset
    dataset_path = "data/"
    
    # Load and preprocess data
    X, y = load_ravdess_data(dataset_path)
    
    if len(X) == 0:
        print("No data found. Please check your dataset path.")
        return
    
    print(f"Loaded {len(X)} audio samples")
    print(f"Number of features: {X.shape[1]}")
    print(f"Emotion classes found: {np.unique([item[0] for item in y])}")
    print(f"Intensity classes found: {np.unique([item[1] for item in y])}")
    
    # Combine emotion and intensity into single labels (emotion * 10 + intensity)
    y_combined = [int(item[0]) * 10 + int(item[1]) for item in y]
    
    # Create label mapping
    emotion_labels = {
        1: "Neutral", 2: "Calm", 3: "Happy", 4: "Sad",
        5: "Angry", 6: "Fearful", 7: "Disgust", 8: "Surprised"
    }
    intensity_labels = {1: "Normal", 2: "Strong"}
    
    combined_labels = {}
    for emotion_code, emotion_name in emotion_labels.items():
        for intensity_code, intensity_name in intensity_labels.items():
            combined_labels[emotion_code * 10 + intensity_code] = f"{emotion_name} ({intensity_name})"
    
    # Encode combined labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_combined)
    
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"Training samples: {X_train.shape[0]}")
    print(f"Testing samples: {X_test.shape[0]}")
    
    # Train Random Forest classifier
    print("\nTraining Random Forest classifier...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
        max_depth=25,
        min_samples_split=2,
        class_weight='balanced'
    )
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_rf = rf_model.predict(X_test)
    accuracy_rf = accuracy_score(y_test, y_pred_rf)
    
    print(f"Random Forest Accuracy: {accuracy_rf:.4f}")
    print("\nClassification Report:")
    
    # Get original class names for display
    class_names = [combined_labels[code] for code in label_encoder.classes_]
    print(classification_report(y_test, y_pred_rf, target_names=class_names))
    
    # Save the model and label encoder
    os.makedirs('models', exist_ok=True)
    
    with open('models/emotion_model.pkl', 'wb') as f:
        pickle.dump(rf_model, f)
    
    with open('models/label_encoder.pkl', 'wb') as f:
        pickle.dump(label_encoder, f)
    
    with open('models/combined_labels.pkl', 'wb') as f:
        pickle.dump(combined_labels, f)
    
    print("Model training completed and saved successfully!")
    
    return rf_model, label_encoder, combined_labels, accuracy_rf

if __name__ == "__main__":
    # Train the model
    model, label_encoder, combined_labels, accuracy = train_emotion_model()
    
    print("\nModel training complete!")
    print(f"Final accuracy: {accuracy:.4f}")
  