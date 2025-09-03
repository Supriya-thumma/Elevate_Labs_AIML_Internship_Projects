
# Human Emotion Detection from Voice 🔊

A machine learning project that detects human emotions from voice recordings using the **RAVDESS dataset**. This system can identify **8 different emotions** with **2 intensity levels** through audio analysis and provides a **Streamlit web interface** for real-time predictions.

![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)
![Librosa](https://img.shields.io/badge/Librosa-Audio_Processing-green.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-Web_UI-red.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-ML_Model-orange.svg)

---

## 🎯 Features

- **Emotion Detection**: Identifies 8 emotions – Neutral, Calm, Happy, Sad, Angry, Fearful, Disgust, Surprised  
- **Intensity Analysis**: Detects both Normal and Strong emotional intensities  
- **Web Interface**: Built using Streamlit for easy interaction  
- **Real-time Processing**: Instant audio analysis and prediction  
- **Comprehensive Features**: MFCC, Chroma, Spectral, and Tonal features  
- **Professional Dataset**: Trained on **RAVDESS dataset** with professional actors  

---

## 📁 Project Structure

```
Human_Emotion_Detection_from_Voice/
├── data/                 # RAVDESS dataset
├── models/               # Trained model files
│   ├── emotion_model.pkl
│   ├── label_encoder.pkl
│   └── combined_labels.pkl
├── main.py               # Model training script
├── app.py                # Streamlit web application
├── utils.py              # Utility functions and feature extraction
├── requirements.txt      # Python dependencies
├── README.md             # Project documentation
└── LICENSE               # MIT License
```

---

## 🚀 Quick Start

### ✅ Prerequisites
- Python 3.9 or higher
- RAVDESS dataset (download from Zenodo)

### ✅ Installation
1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/Human_Emotion_Detection_from_Voice.git
   cd Human_Emotion_Detection_from_Voice
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Download RAVDESS Dataset**
   - [Download from Zenodo](https://zenodo.org/record/1188976)
   - Extract `Audio_Speech_Actors_01-24.zip` into the `data/` directory (or use the dataset files already present in the folder)

4. **Train the Model**
   ```bash
   python main.py
   ```

5. **Run the Web Application**
   ```bash
   streamlit run app.py
   ```

---

## 📊 Dataset Information

The **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset contains:
- 24 professional actors (12 male, 12 female)
- 8 emotional categories with 2 intensity levels
- Audio-only modality files
- Speech content (not song)

### **File Naming Convention**
```
03-01-01-01-01-01-01.wav
│  │  │  │  │  │  └── Actor (01-24)
│  │  │  │  │  └── Repetition (01-02)
│  │  │  │  └── Statement (01-02)
│  │  │  └── Intensity (01=normal, 02=strong)
│  │  └── Emotion (01-08)
│  └── Vocal channel (01=speech, 02=song)
└── Modality (03=audio-only)
```

### **Emotion Mapping**
- 01 = Neutral  
- 02 = Calm  
- 03 = Happy  
- 04 = Sad  
- 05 = Angry  
- 06 = Fearful  
- 07 = Disgust  
- 08 = Surprised  

---

## 🛠️ Technical Details

### **Feature Extraction**
- **MFCC (40 coefficients)**  
- **Chroma Features**  
- **Spectral Contrast**  
- **Tonnetz Features**  
- **RMS energy, Zero-crossing rate, Spectral centroid**  

### **Machine Learning Model**
- Algorithm: **Random Forest Classifier**
- Features: 200+ audio characteristics
- Split: 80% training / 20% testing

### **Web Interface**
- Framework: **Streamlit**
- Input: WAV audio file upload
- Output: Emotion, intensity, confidence score

---

## 📈 Example Output
```
Emotion: Happy
Intensity: Strong
Confidence: 92.5%
```

---

## 🔧 Customization

### Add New Emotions
Update `main.py` and retrain:
```python
emotion_labels = {
    1: "Neutral", 2: "Calm", 3: "Happy", 4: "Sad",
    5: "Angry", 6: "Fearful", 7: "Disgust", 8: "Surprised"
}
```

### Add More Features
In `utils.py`:
```python
features.append(librosa.feature.spectral_flatness(y=audio).mean())
```

---

## 🤝 Contributing
1. Fork the project  
2. Create a feature branch (`git checkout -b feature/YourFeature`)  
3. Commit your changes  
4. Push to the branch and open a Pull Request  

---

## 📝 License
This project is licensed under the **MIT License** – see the [LICENSE](LICENSE) file for details.

---

## 🙏 Acknowledgments
- **RAVDESS Dataset**: Ryerson Audio-Visual Database of Emotional Speech and Song  
- **Librosa**: Audio and music processing library  
- **Scikit-learn**: Machine learning in Python  
- **Streamlit**: Web application framework  

---

