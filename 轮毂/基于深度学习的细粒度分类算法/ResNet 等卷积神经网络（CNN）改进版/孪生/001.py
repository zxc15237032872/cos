import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


# 定义CBAM注意力模块
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, 1, bias=False)
        self.relu = nn.ReLU()
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out * x


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1
        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x) * x


class CBAM(nn.Module):
    def __init__(self, in_channels, reduction=16, kernel_size=7):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x


# 定义带有CBAM的ResNet50模型
class ResNet50_CBAM(models.ResNet):
    def __init__(self, num_classes=2):
        super(ResNet50_CBAM, self).__init__(
            block=models.resnet.Bottleneck,
            layers=[3, 4, 6, 3],
            num_classes=num_classes
        )
        # 在每个残差块组后插入CBAM模块
        self.cbam1 = CBAM(256)
        self.cbam2 = CBAM(512)
        self.cbam3 = CBAM(1024)
        self.cbam4 = CBAM(2048)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.cbam1(x)

        x = self.layer2(x)
        x = self.cbam2(x)

        x = self.layer3(x)
        x = self.cbam3(x)

        x = self.layer4(x)
        x = self.cbam4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x


# 加载预训练的ResNet50模型
from torchvision.models import ResNet50_Weights

model = ResNet50_CBAM(num_classes=2)
pretrained_model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
pretrained_dict = pretrained_model.state_dict()
model_dict = model.state_dict()

# 过滤掉全连接层的权重
pretrained_dict = {k: v for k, v in pretrained_dict.items() if not k.startswith('fc')}
model_dict.update(pretrained_dict)
model.load_state_dict(model_dict)

# 图像预处理
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# 加载并预处理两张图片
image_path1 = '004A.png'
image_path2 = '004A1.png'

image1 = Image.open(image_path1).convert('RGB')
image1 = transform(image1).unsqueeze(0)  # 增加一个batch维度
print(f"Image 1 shape: {image1.shape}")

image2 = Image.open(image_path2).convert('RGB')
image2 = transform(image2).unsqueeze(0)  # 增加一个batch维度
print(f"Image 2 shape: {image2.shape}")

# 将模型设置为评估模式
model.eval()
# 如果只有CPU，将模型和数据移到CPU上
device = torch.device("cpu")
model = model.to(device)
image1 = image1.to(device)
image2 = image2.to(device)

# 提取两张图片的特征向量
with torch.no_grad():
    feature_vector1 = model(image1)
    feature_vector2 = model(image2)

# 输出特征向量
print(f"Image 1 feature vector shape: {feature_vector1.shape}")
print(f"Image 2 feature vector shape: {feature_vector2.shape}")