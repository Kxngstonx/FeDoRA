import torch
import torch.nn as nn
import torch.nn.functional as F

# MobileNetV2의 핵심 구성 요소인 Inverted Residual Block
class InvertedResidual(nn.Module):
    def __init__(self, in_planes, out_planes, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        hidden_dim = int(in_planes * expand_ratio)
        self.use_res_connect = self.stride == 2 and in_planes == out_planes

        layers = []
        if expand_ratio != 1:
            # Point-wise convolution
            layers.append(nn.Conv2d(in_planes, hidden_dim, kernel_size=1, bias=False))
            layers.extend([nn.BatchNorm2d(hidden_dim), nn.ReLU6(inplace=True)])
        
        # Depth-wise convolution
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, stride=stride, padding=1, groups=hidden_dim, bias=False),
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])
        
        # Point-wise convolution
        layers.append(nn.Conv2d(hidden_dim, out_planes, kernel_size=1, bias=False))
        layers.append(nn.BatchNorm2d(out_planes))
        
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)

# MobileNetV2 모델 정의
class MobileNetV2(nn.Module):
    def __init__(self, in_channels=3, num_classes=10, width_mult=1., 
                 use_projection_head=True, proj_out_dim=256):
        super(MobileNetV2, self).__init__()
        self.use_projection_head = use_projection_head
        
        # Inverted Residual Block 설정
        self.cfgs = [
            # t, c, n, s
            [1,  16, 1, 1],
            [6,  24, 2, 2],
            [6,  32, 3, 2],
            [6,  64, 4, 2],
            [6,  96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1],
        ]
        
        # 첫 번째 레이어
        input_channel = int(32 * width_mult)
        self.conv1 = nn.Conv2d(in_channels, input_channel, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(input_channel)
        
        # Inverted Residual Block 생성
        self.layers = self._make_layers(in_planes=input_channel, width_mult=width_mult)
        
        # 마지막 레이어
        last_channel = int(1280 * width_mult) if width_mult > 1.0 else 1280
        self.conv2 = nn.Conv2d(self.cfgs[-1][1], last_channel, kernel_size=1, stride=2, padding=0, bias=False)
        self.bn2 = nn.BatchNorm2d(last_channel)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Classifier (Projection Head 적용)
        if self.use_projection_head:
            self.l1 = nn.Linear(last_channel, last_channel)
            self.l2 = nn.Linear(last_channel, proj_out_dim)
            self.l3 = nn.Linear(proj_out_dim, num_classes)
        else:
            self.fc = nn.Linear(last_channel, num_classes)

        # 가중치 초기화
        self._initialize_weights()

    def _make_layers(self, in_planes, width_mult):
        layers = []
        for i, (t, c, n, s) in enumerate(self.cfgs):
            output_channel = int(c * width_mult)
            for j in range(n):
                stride = s if j == 0 else 1
                layers.append(InvertedResidual(in_planes, output_channel, stride, t))
                in_planes = output_channel
        return nn.Sequential(*layers)
    
    def _forward_impl(self, x):
        x = F.relu6(self.bn1(self.conv1(x)), inplace=True)
        x = self.layers(x)
        x = F.relu6(self.bn2(self.conv2(x)), inplace=True)
        
        x = self.avgpool(x)
        features = torch.flatten(x, 1)

        if self.use_projection_head:
            x_proj = self.l1(features)
            x_proj = F.relu(x_proj)
            x_proj = self.l2(x_proj)
            y = self.l3(x_proj)
            return x_proj, y
        else:
            y = self.fc(features)
            return features, y

    def forward(self, x):
        return self._forward_impl(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
