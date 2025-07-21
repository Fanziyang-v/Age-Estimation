import torch
from torch import nn
from torch.nn import functional as F


class AgeModel(nn.Module):

    def __init__(self, num_classes: int = 101):
        super().__init__()
        # Backbone
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Classifier
        self.conv6 = nn.Conv2d(512, 4096, kernel_size=7)
        self.dropout1 = nn.Dropout(0.5)
        self.conv7 = nn.Conv2d(4096, 4096, kernel_size=1)
        self.dropout2 = nn.Dropout(0.5)
        self.conv8 = nn.Conv2d(4096, num_classes, kernel_size=1)
        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, face_images: torch.Tensor):
        """Predict age probabilities from face images.

        Args:
            face_images (torch.Tensor): face images tensor of shape (B, 3, H, W)

        Returns:
            torch.Tensor: age probabilities tensor of shape (B, num_classes)
        """
        out = F.relu(self.conv1_1(face_images))
        out = F.relu(self.conv1_2(out))
        out = self.pool1(out)
        
        out = F.relu(self.conv2_1(out))
        out = F.relu(self.conv2_2(out))
        out = self.pool2(out)
        
        out = F.relu(self.conv3_1(out))
        out = F.relu(self.conv3_2(out))
        out = F.relu(self.conv3_3(out))
        out = self.pool3(out)
        
        out = F.relu(self.conv4_1(out))
        out = F.relu(self.conv4_2(out))
        out = F.relu(self.conv4_3(out))
        out = self.pool4(out)
        
        out = F.relu(self.conv5_1(out))
        out = F.relu(self.conv5_2(out))
        out = F.relu(self.conv5_3(out))
        out = self.pool5(out)
        
        out = F.relu(self.conv6(out))
        out = self.dropout1(out)
        out = F.relu(self.conv7(out))
        out = self.dropout2(out)
        out = self.conv8(out)
        out = self.flatten(out)
        out = self.softmax(out)
        return out
