import torch
import torch.nn as nn

class TemporalConvNet(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TemporalConvNet, self).__init__()
        # Minimal placeholder implementation
        self.conv1 = nn.Conv1d(num_inputs, num_channels[0], kernel_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.conv1(x)
        out = self.dropout(out)
        return out
