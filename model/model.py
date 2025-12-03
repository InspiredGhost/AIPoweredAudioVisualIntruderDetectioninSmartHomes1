import torch
import torch.nn as nn
import torch.nn.functional as F

class AnomalyClassifier(nn.Module):
    def __init__(self, visual_dim=1000, audio_dim=128, hidden_dim=256, num_classes=2):
        super().__init__()
        self.visual_dim = visual_dim
        self.audio_dim = audio_dim
        
        # Separate processing for visual and audio features
        self.visual_fc = nn.Linear(visual_dim, hidden_dim // 2)
        self.audio_fc = nn.Linear(audio_dim, hidden_dim // 2)
        
        # Combined processing
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout1 = nn.Dropout(0.5)  # Increased dropout for regularization
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.dropout2 = nn.Dropout(0.3)  # Increased dropout for regularization
        self.fc3 = nn.Linear(hidden_dim // 2, num_classes)

    def forward(self, x):
        # Split visual and audio features
        visual_feat = x[:, :self.visual_dim]
        audio_feat = x[:, self.visual_dim:]
        
        # Process separately
        visual_out = F.relu(self.visual_fc(visual_feat))
        audio_out = F.relu(self.audio_fc(audio_feat))
        
        # Combine features
        combined = torch.cat([visual_out, audio_out], dim=1)
        
        # Final processing
        x = F.relu(self.fc1(combined))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.dropout2(x)
        x = self.fc3(x)
        return x
