import streamlit as st
import numpy as np
import librosa
import pickle
import os
import tempfile
from utils import extract_features

# Page configuration
st.set_page_config(
    page_title="Human Emotion Detection from Voice",
    page_icon="🎤",
    layout="wide"
)

# Load the trained model and label information
@st.cache_resource
def load_model():
    try:
        with open('models/emotion_model.pkl', 'rb') as f:
            model = pickle.load(f)
        with open('models/label_encoder.pkl', 'rb') as f:
            label_encoder = pickle.load(f)
        with open('models/combined_labels.pkl', 'rb') as f:
            combined_labels = pickle.load(f)
        return model, label_encoder, combined_labels
    except FileNotFoundError:
        st.error("Model files not found. Please train the model first by running 'python main.py'")
        return None, None, None

model, label_encoder, combined_labels = load_model()

# Emotion color mapping
emotion_colors = {
    "Neutral": "#808080",
    "Calm": "#87CEEB",
    "Happy": "#FFD700",
    "Sad": "#4169E1",
    "Angry": "#FF4500",
    "Fearful": "#8B0000",
    "Disgust": "#006400",
    "Surprised": "#FF69B4"
}

# Main app
st.title("🎤 Human Emotion Detection from Voice")
st.markdown("Upload an audio file (.wav) to detect the emotional content using our trained AI model.")

# File upload section
st.header("📁 Upload Audio File")
audio_file = st.file_uploader("Choose a WAV file", type=["wav"])

if audio_file is not None and model is not None:
    # Display audio information
    st.audio(audio_file, format='audio/wav')
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as tmp_file:
        tmp_file.write(audio_file.getvalue())
        tmp_path = tmp_file.name
    
    try:
        # Extract features and predict emotion
        features = extract_features(tmp_path)
        features = features.reshape(1, -1)
        
        prediction = model.predict(features)
        emotion_code = label_encoder.inverse_transform(prediction)[0]
        
        # Get emotion details
        emotion_label = combined_labels.get(emotion_code, "Unknown")
        
        # Parse emotion and intensity
        emotion_number = emotion_code // 10
        intensity_number = emotion_code % 10
        
        emotion_names = {
            1: "Neutral", 2: "Calm", 3: "Happy", 4: "Sad",
            5: "Angry", 6: "Fearful", 7: "Disgust", 8: "Surprised"
        }
        intensity_names = {1: "Normal", 2: "Strong"}
        
        emotion_name = emotion_names.get(emotion_number, "Unknown")
        intensity_name = intensity_names.get(intensity_number, "Unknown")
        color = emotion_colors.get(emotion_name, "#000000")
        
        # Display results
        st.header("🎯 Prediction Results")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Emotion", emotion_name)
        with col2:
            st.metric("Intensity", intensity_name)
        with col3:
            st.metric("Confidence", f"{np.max(model.predict_proba(features)) * 100:.1f}%")
        
        # Visual display
        st.markdown(f"<div style='background-color:{color}; padding:20px; border-radius:10px; text-align:center;'>"
                   f"<h2 style='color:white;'>{emotion_label}</h2>"
                   f"</div>", unsafe_allow_html=True)
        
        # Additional information
        st.header("ℹ️ Emotion Information")
        
        if emotion_name == "Happy":
            st.info("This emotion is characterized by positive valence, high arousal, and expressions of joy or contentment.")
        elif emotion_name == "Sad":
            st.info("This emotion is characterized by negative valence, low arousal, and expressions of sorrow or disappointment.")
        elif emotion_name == "Angry":
            st.info("This emotion is characterized by negative valence, high arousal, and expressions of frustration or irritation.")
        elif emotion_name == "Fearful":
            st.info("This emotion is characterized by negative valence, high arousal, and expressions of anxiety or apprehension.")
        elif emotion_name == "Surprised":
            st.info("This emotion is characterized by positive valence, high arousal, and expressions of astonishment or amazement.")
        elif emotion_name == "Disgust":
            st.info("This emotion is characterized by negative valence, moderate arousal, and expressions of revulsion or distaste.")
        elif emotion_name == "Neutral":
            st.info("This represents a neutral emotional state with balanced valence and moderate arousal.")
        elif emotion_name == "Calm":
            st.info("This emotion is characterized by positive valence, low arousal, and expressions of peace or relaxation.")
        
    except Exception as e:
        st.error(f"Error processing audio file: {str(e)}")
    finally:
        # Clean up temporary file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)

elif model is None:
    st.warning("Please train the model first by running 'python main.py' in your terminal.")

# Sidebar information
st.sidebar.header("About")
st.sidebar.info("""
This application uses machine learning to detect emotions from voice recordings.
The model was trained on the RAVDESS dataset featuring professional actors.

**Supported Emotions:**
- 😊 Happy
- 😢 Sad 
- 😠 Angry
- 😨 Fearful
- 😲 Surprised
- 🤢 Disgust
- 😐 Neutral
- 😌 Calm

Each emotion can be detected with **Normal** or **Strong** intensity.
""")

st.sidebar.header("Instructions")
st.sidebar.text("""
1. Upload a WAV audio file
2. The model will analyze the audio
3. View the predicted emotion and intensity
4. See additional emotion information
""")
