# VoiceGuard

VoiceGuard is an AI-based voice cloning detection system built with PyTorch and FastAPI. It uses a Convolutional Neural Network (CNN) and MFCC (Mel-frequency cepstral coefficients) to detect whether an audio file is real or AI-generated (fake).

## Setup instructions

1. **Prerequisites**
   Ensure you have Python 3.10+ installed.

2. **Create a virtual environment (optional but recommended)**
   ```bash
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

## Getting the Dataset Ready

We use a Kaggle dataset for voice cloning detection.

Run the provided script to automatically download it:
```bash
python download_kaggle_dataset.py
```

This will create a `dataset` folder with the following structure:
```
voiceguard/
└── dataset/
    ├── REAL/        <-- True/Human .wav audio files
    └── FAKE/        <-- AI Generated .wav audio files
```

## How to Train

To start training the CNN model on your data, simply run:
```bash
python train.py
```

- Training runs for **15 epochs** using an Adam optimizer and CrossEntropy loss.
- Batch size is set to 32.
- After training, the weights are automatically saved to `voiceguard.pth`.

## How to Predict

### Via CLI Script

Use the `predict.py` script provided to check a single `.wav` file:
```bash
python predict.py path/to/your/audio.wav
```
This will print out a label (either `REAL` or `FAKE`) along with a confidence percentage.

### Via FastAPI

An API server is also available:
```bash
uvicorn api:app --reload
```
You can now send a POST request to the API with the audio file.

**Example via cURL:**
```bash
curl -X POST -F "file=@path/to/audio.wav" http://127.0.0.1:8000/predict
```

**Expected JSON Response:**
```json
{
  "label": "REAL",
  "confidence": 0.97
}
```
