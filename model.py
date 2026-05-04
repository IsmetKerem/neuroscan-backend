"""
Multimodal Fusion Model
- EEG branch: Conv1D (14 channels) + Attention -> 128-dim embedding
- Tabular branch: 10 features -> 64-dim embedding
- Classifier: 192 -> 64 -> 3 (Healthy, Alzheimer, Parkinson)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class EEGBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv1d(14, 32, kernel_size=7, padding=3),  # conv.0
            nn.ReLU(),                                      # conv.1
            nn.MaxPool1d(2),                                # conv.2
            nn.Conv1d(32, 64, kernel_size=5, padding=2),  # conv.3
            nn.ReLU(),                                      # conv.4
            nn.AdaptiveAvgPool1d(1),                        # conv.5 (global pool)
        )
        # Attention block
        self.attn = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=1),  # attn.0
            nn.ReLU(),                          # attn.1
            nn.Conv1d(32, 64, kernel_size=1),  # attn.2
            nn.Sigmoid(),
        )
        self.fc = nn.Linear(64, 128)  # fc

    def forward(self, x):
        # x: (B, 14, T)
        feat = self.conv(x)            # (B, 64, 1)
        att = self.attn(feat)          # (B, 64, 1)
        feat = feat * att              # gated attention
        feat = feat.squeeze(-1)        # (B, 64)
        return self.fc(feat)           # (B, 128)


class TabularBranch(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(10, 32),  # net.0
            nn.ReLU(),           # net.1
            nn.Linear(32, 64),   # net.2
            nn.ReLU(),
        )

    def forward(self, x):
        return self.net(x)  # (B, 64)


class MultiModalNet(nn.Module):
    def __init__(self, num_classes=3):
        super().__init__()
        self.eeg = EEGBranch()
        self.tab = TabularBranch()
        self.classifier = nn.Sequential(
            nn.Linear(192, 64),  # classifier.0
            nn.ReLU(),            # classifier.1
            nn.Dropout(0.3),      # classifier.2
            nn.Linear(64, num_classes),  # classifier.3
        )

    def forward(self, eeg, tab):
        e = self.eeg(eeg)                 # (B, 128)
        t = self.tab(tab)                  # (B, 64)
        fused = torch.cat([e, t], dim=1)   # (B, 192)
        return self.classifier(fused)      # (B, 3)


CLASS_NAMES = ["Healthy", "Alzheimer", "Parkinson"]
