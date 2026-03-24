import sys
import torch
import torch.nn.functional as F
from model import VoiceGuardCNN
from dataset import extract_features

def predict(wav_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = VoiceGuardCNN()
    try:
        model.load_state_dict(torch.load("voiceguard.pth", map_location=device))
    except FileNotFoundError:
        print("Error: Model weights not found. Please train the model first by running 'python train.py'.")
        sys.exit(1)
        
    model.to(device)
    model.eval()
    
    # Extract features
    try:
        mfcc = extract_features(wav_path)
    except Exception as e:
        print(f"Error processing {wav_path}: {e}")
        sys.exit(1)
        
    # Shape: (1, 1, 40, 128)
    input_tensor = torch.tensor(mfcc, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)[0]
        
    # 0 = REAL, 1 = FAKE
    prob_fake = probabilities[1].item()
    prob_real = probabilities[0].item()
    
    if prob_fake > 0.5:
        label = "FAKE"
        confidence = prob_fake
    else:
        label = "REAL"
        confidence = prob_real
        
    print(f"File: {wav_path}")
    print(f"Prediction: {label}")
    print(f"Confidence: {confidence * 100:.2f}%")

if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python predict.py <path_to_wav_file>")
        sys.exit(1)
        
    predict(sys.argv[1])
