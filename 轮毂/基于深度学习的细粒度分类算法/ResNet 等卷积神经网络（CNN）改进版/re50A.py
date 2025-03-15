import numpy as np
import torch
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image


def reshape_feature_to_image(feature):
    # 这里需要根据实际特征向量的维度进行调整
    size = int(np.sqrt(len(feature)))
    return feature.reshape((size, size))


# 加载预训练的ResNet - 50模型，使用weights参数
model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
# 去掉最后一层全连接层，用于特征提取
model = torch.nn.Sequential(*list(model.children())[:-1])
model.eval()

# 定义图像预处理步骤
preprocess = transforms.Compose([
    transforms.Resize(80),

    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def extract_features(image_path):
    """
    提取图像的特征向量
    """
    # 打开图像
    image = Image.open(image_path).convert('RGB')
    # 预处理图像
    input_tensor = preprocess(image)
    input_batch = input_tensor.unsqueeze(0)

    # 确保使用CPU
    input_batch = input_batch.to('cpu')
    model.to('cpu')

    # 提取特征
    with torch.no_grad():
        features = model(input_batch)
    # 将特征向量转换为numpy数组
    features = features.squeeze().cpu().numpy()
    return features


def cosine_similarity(feature1, feature2):
    """
    计算两个特征向量的余弦相似度
    """
    dot_product = np.dot(feature1, feature2)
    norm_feature1 = np.linalg.norm(feature1)
    norm_feature2 = np.linalg.norm(feature2)
    similarity = dot_product / (norm_feature1 * norm_feature2)
    return similarity


# 定义三组图片路径
image_pairs = [
    ('006A.png', '006A1.png'),
    ('004A.png', '004A1.png'),
    ('005A.png', '005B.png'),
    ('006A.png', '004A1.png'),
    ('004A.png', '005A.png'),
    ('005A.png', '006A.png')
]

# 循环处理每组图片
for i, (image_path1, image_path2) in enumerate(image_pairs, start=1):
    feature1 = extract_features(image_path1)
    max_similarity = -1
    best_angle = 0
    for angle in range(2, 101):
        rotated_image = Image.open(image_path2).convert('RGB').rotate(angle)
        input_tensor = preprocess(rotated_image)
        input_batch = input_tensor.unsqueeze(0).to('cpu')
        with torch.no_grad():
            features = model(input_batch)
        features = features.squeeze().cpu().numpy()
        similarity = cosine_similarity(feature1, features)
        if similarity > max_similarity:
            max_similarity = similarity
            best_angle = angle
    print(f"第 {i} 组图片（{image_path1} 和 {image_path2}）的最佳旋转角度: {best_angle} 度")
    rotated_image = Image.open(image_path2).convert('RGB').rotate(best_angle)
    input_tensor = preprocess(rotated_image)
    input_batch = input_tensor.unsqueeze(0).to('cpu')
    with torch.no_grad():
        feature2 = model(input_batch).squeeze().cpu().numpy()
    final_similarity = cosine_similarity(feature1, feature2)
    print(f"第 {i} 组图片（{image_path1} 和旋转后的 {image_path2}）的最终相似度: {final_similarity}")