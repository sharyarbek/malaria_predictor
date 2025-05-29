from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

import io 
from PIL import Image

''' CNN things '''
import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision import transforms

app = FastAPI(
    title='Malaria Detector'
)

# if gpu exists -> GPU resources are utilized; otherwise CPU resources are utilized
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# NeuralNet
class NeuralNet(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3,32,5) # (128 - 5 ) * 3
        self.bn1 = nn.BatchNorm2d(32)
        self.pool =  nn.MaxPool2d(3, 3)
        
        self.conv2 = nn.Conv2d(32, 64, 5)
        self.bn2 = nn.BatchNorm2d(64)
        
        ''' из-за оверфиттинга минусуем слой '''
        # self.conv3 = nn.Conv2d(64, 128, 5)
        # self.bn3 = nn.BatchNorm2d(128)
        
        self.dropout = nn.Dropout(p=0.5)
        
        self.fc1 = nn.Linear(64 * 12 * 12, 120)  # 64 instead of 128, also changing 2 to 12
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)
        
    def forward(self, x):
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        # x = self.pool(F.relu(self.bn3(self.conv3(x))))
        x = torch.flatten(x, 1)
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.fc3(x)
        return x


# Config
MODEL_PATH = 'best_trained_net.pth'
IMG_SIZE = 128
CLASS_NAMES = ['Parasited', 'Uninfected']


# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5]*3, std=[0.5]*3)
])


# loading our model
net = NeuralNet()
net.load_state_dict(torch.load(MODEL_PATH, map_location=device))
net.to(device)
net.eval()


@app.get('/')
def root_page():
    return {'message' : 'Abnormality Detection in Medical Images Using CNN'}

@app.post('/predictor')
async def predictor(file: UploadFile = File(...)):
    if file.content_type not in ('image/png', 'image/jpeg'):
        raise HTTPException(415, detail='Unsupported file type')
    
    # Image reading
    data = await file.read()
    img = Image.open(io.BytesIO(data)).convert('RGB')

    # Tensor preperation
    x = preprocess(img).unsqueeze(0).to(device)
    
    # inference
    with torch.no_grad():
        logits = net(x)
        probs = F.softmax(logits, dim=1).cpu().numpy()[0]
    
    # Prediction stage
    idx = probs.argmax()
    return JSONResponse({
        "class": CLASS_NAMES[idx],
        "probability": float(probs[idx])
    })