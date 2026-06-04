import streamlit as st
import numpy as np
import cv2
from PIL import Image
import time
import pandas as pd
import torch
import torch.nn as nn

# ── Model definition ──────────────────────────────
class LWDCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 4, 3, padding=0, bias=False)
        self.bn1   = nn.BatchNorm2d(4)
        self.pool  = nn.MaxPool2d(3, 3)
        self.conv2 = nn.Conv2d(4, 4, 3, padding=0, bias=False)
        self.bn2   = nn.BatchNorm2d(4)
        self.conv3 = nn.Conv2d(4, 8, 3, padding=0, bias=False)
        self.bn3   = nn.BatchNorm2d(8)
        self.dense1 = nn.Linear(72, 16)
        self.bn4    = nn.BatchNorm1d(16)
        self.dense2 = nn.Linear(16, 1)
        self.relu   = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.pool(self.relu(self.bn1(self.conv1(x))))
        x = self.pool(self.relu(self.bn2(self.conv2(x))))
        x = self.pool(self.relu(self.bn3(self.conv3(x))))
        x = x.view(x.size(0), -1)
        x = self.relu(self.bn4(self.dense1(x)))
        x = self.sigmoid(self.dense2(x))
        return x
