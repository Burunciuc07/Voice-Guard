import os
import shutil
import torch
import torch.nn.functional as F
from fastapi import FastAPI, UploadFile, File, HTTPException
from model import VoiceGuardCNN
from dataset import extract_features

app = FastAPI(title="VoiceGuard API", description="AI-based voice cloning detection system")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VoiceGuardCNN()
model_loaded = False

if os.path.exists("voiceguard.pth"):
    model.load_state_dict(torch.load("voiceguard.pth", map_location=device))
    model.to(device)
    model.eval()
    model_loaded = True

@app.post("/predict")
async def predict_endpoint(file: UploadFile = File(...)):
    global model_loaded
    
    if not file.filename.endswith(".wav"):
        raise HTTPException(status_code=400, detail="Only .wav files are supported")
        
    if not model_loaded:
        # Try loading again in case the server was started before compiling/training
        if os.path.exists("voiceguard.pth"):
            model.load_state_dict(torch.load("voiceguard.pth", map_location=device))
            model.to(device)
            model.eval()
            model_loaded = True
        else:
            raise HTTPException(status_code=503, detail="Model weights not found. Train the model first using 'python train.py'.")
        
    temp_file_path = f"temp_{file.filename}"
    with open(temp_file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        mfcc = extract_features(temp_file_path)
    except Exception as e:
        os.remove(temp_file_path)
        raise HTTPException(status_code=500, detail=f"Error processing audio: {e}")
        
    os.remove(temp_file_path)
    
    input_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        
    prob_fake = probabilities[1].item()
    prob_real = probabilities[0].item()
    
    if prob_fake > 0.5:
        return {"label": "FAKE", "confidence": prob_fake}
    else:
        return {"label": "REAL", "confidence": prob_real}
