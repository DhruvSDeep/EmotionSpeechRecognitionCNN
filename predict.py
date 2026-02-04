import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import librosa
import wave           #includes some unused imports, I was trying new things
import os
import pylab
import numpy as np
from sklearn.model_selection import train_test_split
import torchvision
import torch
from torch.utils.data import DataLoader, random_split
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.io import read_image
import torch.optim
import tqdm
import torch.nn.functional as F
import torchmetrics
from torchmetrics import Accuracy
from PIL import Image
import torch.amp


class CNN(nn.Module):             #defining our model and forward prop
    def __init__(self, in_channels, num_classes):
 
        super(CNN, self).__init__()

        self.pool = nn.MaxPool2d(kernel_size=2)

        self.conv1 = nn.Conv2d(in_channels=in_channels, out_channels=32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)

        self.conv3 = nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(128)

        self.conv4 = nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3,padding=1)
        self.bn4 = nn.BatchNorm2d(128)

        #self.fc1 = nn.Linear(128 * 30 * 40, 256)#for no gap/gmp
        self.fc1 = nn.Linear(128 * 2, 512)       #for gap/gmp
        self.fc2 = nn.Linear(512, num_classes)

        self.gap1 = nn.AdaptiveAvgPool2d(1)
        self.gmp1 = nn.AdaptiveMaxPool2d(1)

        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout2d(0.2)


    def forward(self, x):
   
        x = F.relu(self.bn1(self.conv1(x)))  
        x = self.pool(x) 

        x = F.relu(self.bn2(self.conv2(x)))  
        x = self.pool(x)

        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool(x)
        x = self.dropout2(x)         


        x = F.relu(self.bn4(self.conv4(x)))
        x = self.pool(x)
        x = self.dropout2(x)         

        
        xGap = self.gap1(x)
        xGmp = self.gmp1(x)
        x = torch.cat((xGap, xGmp), dim=1)  
        x = torch.flatten(x, 1)


        x = F.relu(self.fc1(x))  
        x = self.dropout1(x)
        x = self.fc2(x)         


        return x


nameFile = input("Enter name for audio file with extension")
y, sr = librosa.load(os.path.join(f'./{nameFile}'), sr=22050)
pylab.axis('off')
pylab.axes([0., 0., 1., 1.], frameon=False, xticks=[], yticks=[])
S = librosa.feature.melspectrogram(y=y, sr=sr)
librosa.display.specshow(librosa.power_to_db(S, ref=np.max), vmin=-80, vmax=0, sr=sr)
pylab.savefig(f"./{nameFile}_Spectrogram.jpg", bbox_inches=None, pad_inches=0)
pylab.close()
pylab.clf()

device = 'cpu'
if torch.cuda.is_available():
    device = 'cuda'

model = CNN(in_channels=3, num_classes=8)
model.load_state_dict(torch.load('bestModel.pth', weights_only=True))
model.to(device)

transform = transforms.Compose([transforms.ToTensor()])
imag = Image.open(f"{nameFile}_Spectrogram.jpg")
tensor = transform(imag)
out = model(tensor.unsqueeze(0).to(device))

probs = torch.softmax(out, dim=1)
conf, pred = torch.max(probs, dim=1)

emotionMap = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
    }

print(f"prediction : {emotionMap[str(pred.item()+1).zfill(2)]}, confidence: {conf.item()*100:.2f}%")