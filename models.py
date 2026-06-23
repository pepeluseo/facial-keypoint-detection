## Define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


class Net(nn.Module):
    """
    Convolutional Neural Network for Facial Keypoint Detection.

    Input:
        Grayscale image tensor of shape: (batch_size, 1, 224, 224)

    Output:
        136 values per image:
        68 facial keypoints * 2 coordinates (x, y)
    """

    def __init__(self):
        super(Net, self).__init__()

        # Convolutional layers
        # Input image: 1 x 224 x 224
        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=5)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)

        # Max pooling layer
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Dropout to reduce overfitting
        self.dropout = nn.Dropout(p=0.25)

        # Fully connected layers
        # After conv/pool layers, image size becomes:
        # 224 -> conv1 220 -> pool 110
        # 110 -> conv2 106 -> pool 53
        # 53 -> conv3 49 -> pool 24
        # 24 -> conv4 22 -> pool 11
        # Final tensor: 128 x 11 x 11
        self.fc1 = nn.Linear(128 * 11 * 11, 500)
        self.fc2 = nn.Linear(500, 136)

        # Initialize weights
        self._initialize_weights()

    def forward(self, x):
        # Convolution + ReLU + Pooling
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        x = self.pool(F.relu(self.conv4(x)))

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully connected layers
        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        # Final output: 136 values
        x = self.fc2(x)

        return x

    def _initialize_weights(self):
        """
        Initialize weights using Xavier initialization for better training stability.
        """
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                I.xavier_uniform_(module.weight)
                if module.bias is not None:
                    I.constant_(module.bias, 0)

            elif isinstance(module, nn.Linear):
                I.xavier_uniform_(module.weight)
                if module.bias is not None:
                    I.constant_(module.bias, 0)