# Audio Training Script
# Trains a classifier for gunshot sounds in data/audio/gunshot/

import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchaudio
from torch.utils.data import DataLoader, Dataset

DATA_DIR = '../../data/audio/gunshot/'
MODEL_PATH = '../../model/audio_model.pth'

class GunshotDataset(Dataset):
    def __init__(self, data_dir):
        self.files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.wav')]
    def __len__(self):
        return len(self.files)
    def __getitem__(self, idx):
        waveform, sr = torchaudio.load(self.files[idx])
        # Extract Mel spectrogram
        mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sr)(waveform)
        mel_spec = mel_spec.squeeze(0)
        label = 1  # gunshot
        return mel_spec, label

train_dataset = GunshotDataset(DATA_DIR)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

class SimpleAudioCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, 3, padding=1)
        self.pool = nn.MaxPool2d(2,2)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.fc1 = nn.Linear(32*32*32, 64)
        self.fc2 = nn.Linear(64, 2) # gunshot vs not-gunshot
    def forward(self, x):
        x = x.unsqueeze(1) # (batch, 1, mel, time)
        x = self.pool(torch.relu(self.conv1(x)))
        x = self.pool(torch.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleAudioCNN()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

for epoch in range(5):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = torch.tensor(labels).to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')

# Save model
torch.save(model.state_dict(), MODEL_PATH)
print('Audio model trained and saved.')
