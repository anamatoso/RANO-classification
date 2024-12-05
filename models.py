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


# ResNet
def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False)

def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = func.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=3, stride=stride, dilation=dilation, padding=dilation, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dilation = dilation

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class ResNet(nn.Module):

    def __init__(self,
                 block,
                 layers,
                 sample_input_D,
                 sample_input_H,
                 sample_input_W,
                 num_seg_classes,
                 shortcut_type='B',
                 no_cuda=False):
        self.sample_input_W = sample_input_W
        self.sample_input_H = sample_input_H
        self.sample_input_D = sample_input_D
        self.inplanes = 64
        self.no_cuda = no_cuda
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv3d(
            1,
            64,
            kernel_size=7,
            stride=(2, 2, 2),
            padding=(3, 3, 3),
            bias=False)

        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 128, layers[1], shortcut_type, stride=2)
        self.layer3 = self._make_layer(
            block, 256, layers[2], shortcut_type, stride=1, dilation=2)
        self.layer4 = self._make_layer(
            block, 512, layers[3], shortcut_type, stride=1, dilation=4)

        self.conv_seg = nn.Sequential(
            nn.ConvTranspose3d(
                512 * block.expansion,
                32,
                2,
                stride=2
            ),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                32,
                32,
                kernel_size=3,
                stride=(1, 1, 1),
                padding=(1, 1, 1),
                bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
            nn.Conv3d(
                32,
                num_seg_classes,
                kernel_size=1,
                stride=(1, 1, 1),
                bias=False)
        )

        self.classifier = nn.Sequential(
            nn.Conv3d(4, 1, kernel_size=4, stride=2, padding=1, bias=False),  # 4 x 30 x 30 x 20
            nn.ReLU(),
            nn.AvgPool3d(kernel_size=2),  # 1 x 15 x 15 x 10
            nn.Flatten(),
            nn.Linear(2250, num_seg_classes))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False), nn.BatchNorm3d(planes * block.expansion))

        layers = [block(self.inplanes, planes, stride=stride, dilation=dilation, downsample=downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv_seg(x)
        x = self.classifier(x)

        return x

def resnet18(**kwargs):
    """Constructs a ResNet-18 model.
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    return model


# Densenet with clinical data as input
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
