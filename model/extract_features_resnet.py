import os
import numpy as np
import cv2
import json
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm

# Load label map
with open(os.path.join(os.path.dirname(__file__), '../config/label_map.json')) as f:
    label_map = json.load(f)

video_dir = os.path.join(os.path.dirname(__file__), '../data/video')
features = []
labels = []

# Pretrained ResNet18 feature extractor
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet = torch.nn.Sequential(*list(resnet.children())[:-1])
resnet.eval()
transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

import pickle
progress_path = os.path.join(os.path.dirname(__file__), '../model/extract_progress_resnet.pkl')
try:
    with open(progress_path, 'rb') as pf:
        processed_files = pickle.load(pf)
except Exception:
    processed_files = set()

for class_name, class_idx in label_map.items():
    if class_name == 'normal':
        continue
    class_folder = os.path.join(video_dir, class_name)
    if not os.path.isdir(class_folder):
        continue
    for fname in tqdm(os.listdir(class_folder), desc=class_name):
        if not fname.lower().endswith(('.mp4', '.avi', '.mov')):
            continue
        video_path = os.path.join(class_folder, fname)
        if video_path in processed_files:
            continue
        cap = cv2.VideoCapture(video_path)
        frame_features = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            input_tensor = transform(frame).unsqueeze(0)
            with torch.no_grad():
                feat = resnet(input_tensor).view(-1).numpy()
            frame_features.append(feat)
        cap.release()
        if frame_features:
            video_feat = np.mean(frame_features, axis=0)
            features.append(video_feat)
            labels.append(class_idx)
        processed_files.add(video_path)
        with open(progress_path, 'wb') as pf:
            pickle.dump(processed_files, pf)

features = np.array(features)
labels = np.array(labels)
np.savez(os.path.join(os.path.dirname(__file__), '../model/features.npz'), visual=features, audio=np.zeros((features.shape[0], 0)), labels=labels)
print('ResNet18 features extracted and saved to model/features.npz')
