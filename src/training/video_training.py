# Video Training Script
# Trains a classifier for video anomalies in data/video/

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import numpy as np
import cv2
import numpy as np
from tqdm import tqdm

# Paths
TRAIN_CSV = os.path.join('data/video', 'train.csv')
DATA_DIR = 'data/video'
MODEL_PATH = 'model/video_model.pth'

# Data transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# Custom dataset loader (placeholder)
class VideoDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, data_dir, transform=None):
        import pandas as pd
        self.data = pd.read_csv(csv_file)
        self.data_dir = data_dir
        self.transform = transform
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        video_path = row['video_path']
        import cv2
        cap = cv2.VideoCapture(video_path)
        frames = []
        frame_count = 0
        SEQ_LEN = 16  # Number of frames per sequence
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % 5 == 0:  # every 5th frame
                if self.transform:
                    from PIL import Image
                    frame = self.transform(Image.fromarray(frame))
                frames.append(frame)
            frame_count += 1
            if len(frames) == SEQ_LEN:
                break
        cap.release()
        # Pad if not enough frames
        if len(frames) < SEQ_LEN:
            import numpy as np
            pad_frame = self.transform(Image.fromarray(np.zeros((224,224,3), dtype=np.uint8)))
            frames += [pad_frame] * (SEQ_LEN - len(frames))
        label = row['label']
        # Return sequence and label
        return torch.stack(frames), label

def video_collate_fn(batch):
    # batch: list of (frames, label)
    frames_list, labels_list = zip(*batch)
    return frames_list, labels_list

# Dataset and loader
train_dataset = VideoDataset(TRAIN_CSV, DATA_DIR, transform)
train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True, collate_fn=video_collate_fn)

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
from model.inference import EnhancedAnomalyDetector

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Feature extractor (ResNet18 without final layer)
feature_extractor = models.resnet18(pretrained=True)
feature_extractor = nn.Sequential(*list(feature_extractor.children())[:-1])  # Remove final FC
feature_extractor.eval()
feature_extractor.to(device)

# Sequence model (TCN-based)
num_classes = len(os.listdir(DATA_DIR))
visual_dim = 512  # ResNet18 output
model = EnhancedAnomalyDetector(visual_dim=visual_dim, audio_dim=0, hidden_dim=256, num_classes=num_classes, sequence_length=16)
model.to(device)

# Training setup
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# In training loop, update label indexing:
train_labels = train_dataset.data['label'].unique().tolist()

# Set this to resume from a specific video index if needed
start_video_idx = 0

# Load checkpoint if exists
checkpoint_path = MODEL_PATH + '.checkpoint'
if os.path.exists(checkpoint_path):
    print(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    # optimizer.load_state_dict(checkpoint['optimizer_state_dict'])  # Disabled: incompatible parameter groups
    start_video_idx = checkpoint['video_idx'] + 1
    print(f"Resuming from video {start_video_idx}")
else:
    print("No checkpoint found, starting from scratch.")

# Training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    batch_iter = tqdm(train_loader, desc=f"Epoch {epoch+1}")
    total_videos = len(train_dataset)
    for batch_idx, (frames_list, labels_list) in enumerate(batch_iter):
        batch_loss = 0.0
        for video_idx, (frames, label) in enumerate(zip(frames_list, labels_list)):
            global_video_idx = batch_iter.n * train_loader.batch_size + video_idx
            if global_video_idx < start_video_idx:
                continue  # skip already processed videos
            video_path = train_dataset.data.iloc[global_video_idx]['video_path']
            print(f"Processing video {global_video_idx+1} of {total_videos}: {video_path}")
            # Extract features for each frame in the sequence
            frame_features = []
            for frame_idx, frame in enumerate(frames):
                with torch.no_grad():
                    frame = frame.unsqueeze(0).to(device)  # (1, C, H, W)
                    feat = feature_extractor(frame)  # (1, 512, 1, 1)
                    feat = feat.view(512)  # Flatten
                frame_features.append(feat)
                # Frame visualization removed for training speed
            # Sliding window over all frames
            window_size = 16  # or your model's expected sequence length
            stride = 1
            num_windows = max(1, len(frame_features) - window_size + 1)
            for win_start in range(0, num_windows, stride):
                win_end = win_start + window_size
                window_feats = frame_features[win_start:win_end]
                if len(window_feats) < window_size:
                    # Pad if not enough frames
                    pad = [torch.zeros_like(window_feats[0])]*(window_size-len(window_feats))
                    window_feats += pad
                visual_seq = torch.stack(window_feats).unsqueeze(0).to(device)  # (1, window_size, 512)
                # No audio features for now
                audio_seq = torch.zeros((1, window_size, 0)).to(device)
                labels = torch.tensor([train_labels.index(label)]).to(device)
                optimizer.zero_grad()
                outputs = model(visual_seq, audio_seq)
                logits = outputs['logits']
                loss = criterion(logits, labels)
                loss.backward()
                optimizer.step()
                batch_loss += loss.item()
                print(f"Epoch {epoch+1}/{num_epochs}, Batch {batch_idx+1}, Video {global_video_idx+1}/{total_videos}, Window {win_start+1}/{num_windows}, Loss: {loss.item():.4f}")
            # Close all OpenCV windows after each video
            # Frame visualization removed for training speed
            # No audio features for now
            audio_seq = torch.zeros((1, 16, 0)).to(device)
            labels = torch.tensor([train_labels.index(label)]).to(device)
            optimizer.zero_grad()
            outputs = model(visual_seq, audio_seq)
            logits = outputs['logits']
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            batch_loss += loss.item()
            print(f"Epoch {epoch+1}, Batch {batch_idx+1}, Video {global_video_idx+1}/{total_videos}, Loss: {loss.item():.4f}")
            # Save checkpoint after each video
            torch.save({
                'epoch': epoch,
                'video_idx': global_video_idx,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, MODEL_PATH + '.checkpoint')
            print(f"Checkpoint saved after video {global_video_idx+1}.")
        running_loss += batch_loss
        batch_iter.set_postfix(loss=batch_loss)
    print(f'Epoch {epoch+1}, Loss: {running_loss/len(train_loader):.4f}')
    torch.save(model.state_dict(), MODEL_PATH)
    print(f'Model weights saved after epoch {epoch+1}.')

# Save model
torch.save(model.state_dict(), MODEL_PATH)
print('Video model trained and saved.')
