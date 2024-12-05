"""Models file

"""

import torch
import torch.nn as nn
from functools import partial
import torch.nn.functional as func
from torch.autograd import Variable


# 3D version of AlexNet
class AlexNet3D(nn.Module):
    def __init__(self, num_channels=2,num_classes=4):
        super(AlexNet3D, self).__init__()
        
        self.features = nn.Sequential(
            nn.Conv3d(num_channels, 64, kernel_size=11, stride=4, padding=(0, 2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            
            nn.Conv3d(64, 192, kernel_size=5, padding=(0, 2, 2)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
            
            nn.Conv3d(192, 384, kernel_size=3, padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(384, 256, kernel_size=3, padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            
            nn.Conv3d(256, 256, kernel_size=3, padding=(0, 1, 1)),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=3, stride=2),
        )
        
        self.avgpool = nn.AdaptiveAvgPool3d((6, 6, 6))

        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(256 * 6 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
            # nn.Softmax()
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

class DenseNetWithClinical(nn.Module):
    def __init__(self, densenet_model, num_classes, clinical_data_dim, hidden_dim=256):
        super(DenseNetWithClinical, self).__init__()
        
        # Initialize DenseNet (without its classification layer)
        self.densenet = densenet_model
        # Remove the last fully connected (classification) layer of DenseNet
        self.densenet.class_layers.out = nn.Identity()

        # Define a fully connected layer to process clinical data
        self.clinical_fc = nn.Sequential(
            nn.Linear(clinical_data_dim, hidden_dim),
            nn.ReLU()
        )
        
        # Define the fully connected layer that concatenates both features
        densenet_output_dim = 2688  # For DenseNet264, output after pooling is 2688
        combined_dim = densenet_output_dim + hidden_dim
        self.final_fc = nn.Sequential(
            nn.Linear(combined_dim, num_classes)
        )

    def forward(self, image, clinical_data):
        # Forward pass through DenseNet for image features
        image_features = self.densenet(image)
        
        # Forward pass through clinical data processing layers
        clinical_features = self.clinical_fc(clinical_data)
        
        # Concatenate both image and clinical features
        combined_features = torch.cat((image_features, clinical_features), dim=1)
        
        # Final classification layer
        output = self.final_fc(combined_features)
        return output
    
class Inception3DModule(nn.Module):
    def __init__(self, in_channels, out1x1, red3x3, out3x3, red5x5, out5x5, pool_proj):
        super(Inception3DModule, self).__init__()

        # 1x1 Convolution
        self.branch1x1 = nn.Conv3d(in_channels, out1x1, kernel_size=1)

        # 1x1 Convolution followed by 3x3 Convolution
        self.branch3x3 = nn.Sequential(
            nn.Conv3d(in_channels, red3x3, kernel_size=1),
            nn.Conv3d(red3x3, out3x3, kernel_size=3, padding=1)
        )

        # 1x1 Convolution followed by 5x5 Convolution
        self.branch5x5 = nn.Sequential(
            nn.Conv3d(in_channels, red5x5, kernel_size=1),
            nn.Conv3d(red5x5, out5x5, kernel_size=5, padding=2)
        )

        # 3x3 Max Pooling followed by 1x1 Convolution
        self.branch_pool = nn.Sequential(
            nn.MaxPool3d(kernel_size=3, stride=1, padding=1),
            nn.Conv3d(in_channels, pool_proj, kernel_size=1)
        )

    def forward(self, x):
        branch1x1 = self.branch1x1(x)
        branch3x3 = self.branch3x3(x)
        branch5x5 = self.branch5x5(x)
        branch_pool = self.branch_pool(x)

        outputs = [branch1x1, branch3x3, branch5x5, branch_pool]
        return torch.cat(outputs, 1)

class GoogleNet3D(nn.Module):
    def __init__(self, num_channels=2, num_classes=4):
        super(GoogleNet3D, self).__init__()

        # Initial Convolution
        self.conv1_v2 = nn.Conv3d(num_channels, 64, kernel_size=7, stride=2, padding=3)#,groups=num_channels)
        self.max_pool1 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Inception Modules
        self.inception1 = Inception3DModule(64, 64, 64, 128, 32, 32, 32)
        self.inception2 = Inception3DModule(256, 128, 128, 192, 96, 96, 64)

        # Downsample
        self.max_pool2 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Inception Modules
        self.inception3 = Inception3DModule(480, 192, 96, 208, 16, 48, 64)
        self.inception4 = Inception3DModule(512, 160, 112, 224, 24, 64, 64)
        self.inception5 = Inception3DModule(512, 128, 128, 256, 24, 64, 64)
        self.inception6 = Inception3DModule(512, 112, 144, 288, 32, 64, 64)
        self.inception7 = Inception3DModule(528, 256, 160, 320, 32, 128, 128)

        # Downsample
        self.max_pool3 = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        # Inception Modules
        self.inception8 = Inception3DModule(832, 256, 160, 320, 32, 128, 128)
        self.inception9 = Inception3DModule(832, 384, 192, 384, 48, 128, 128)

        # Global Average Pooling
        self.avg_pool = nn.AdaptiveAvgPool3d(1)

        # Fully Connected Layer
        self.fc = nn.Linear(1024, num_classes)

        # Softmax
        # self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = func.relu(self.conv1_v2(x))
        x = self.max_pool1(x)

        x = self.inception1(x)
        x = self.inception2(x)

        x = self.max_pool2(x)

        x = self.inception3(x)
        x = self.inception4(x)
        x = self.inception5(x)
        x = self.inception6(x)
        x = self.inception7(x)

        x = self.max_pool3(x)

        x = self.inception8(x)
        x = self.inception9(x)

        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        # x = self.softmax(x)

        return x
