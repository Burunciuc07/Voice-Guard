import torch
from model import VoiceGuardCNN
from dataset import VoiceDataset
from torch.utils.data import DataLoader

try:
    model = VoiceGuardCNN()
    model.load_state_dict(torch.load('voiceguard.pth', map_location='cpu'))
    model.eval()

    ds = VoiceDataset('dataset')
    
    loader = DataLoader(ds, batch_size=32)

    real_correct = 0
    real_total = 0
    fake_correct = 0
    fake_total = 0
    
    print('Evaluating dataset...')
    with torch.no_grad():
        for inputs, labels in loader:
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            
            for p, l in zip(preds, labels):
                if l.item() == 0:
                    real_total += 1
                    if p.item() == 0:
                        real_correct += 1
                else:
                    fake_total += 1
                    if p.item() == 1:
                        fake_correct += 1
                        
            # Stop early for speed
            if real_total + fake_total > 400:
                break

    print(f'REAL acc: {real_correct}/{max(1,real_total)} ({real_correct/max(1,real_total)*100:.2f}%)')
    print(f'FAKE acc: {fake_correct}/{max(1,fake_total)} ({fake_correct/max(1,fake_total)*100:.2f}%)')
except Exception as e:
    print(f'Error: {e}')
