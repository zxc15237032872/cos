import torch
from transformers import AutoImageProcessor, AutoModel
import numpy as np
from huggingface_hub import hf_hub_download
import os
from PIL import Image
import cv2


def extract_dinov2_features(image_path):
    cache_dir = "D:\\soft\\huggingface\\transformers"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    config_file = hf_hub_download(repo_id="facebook/dinov2-base", filename="config.json", cache_dir=cache_dir)
    weights_file = hf_hub_download(repo_id="facebook/dinov2-base", filename="pytorch_model.bin", cache_dir=cache_dir)

    image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", cache_dir=cache_dir)
    model = AutoModel.from_pretrained("facebook/dinov2-base", cache_dir=cache_dir)

    image = Image.open(image_path).convert("RGB")
    img = image_processor(images=image, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**img)
    embedding = outputs.last_hidden_state[:, 0, :].squeeze(1).squeeze(0)
    return embedding.numpy()


def extract_sift_features(image_path):
    img = cv2.imread(image_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create()
    kp, des = sift.detectAndCompute(gray, None)
    if des is None:
        des = np.array([])
    return des.flatten()  # 将特征描述符展平为一维向量


def calculate_euclidean_distance(vec1, vec2):
    distance = np.linalg.norm(vec1 - vec2)
    return distance


# 提取两张图片的 DINOv2 特征
image1_dinov2_features = extract_dinov2_features('006A1.png')
image2_dinov2_features = extract_dinov2_features('006A.png')

# 提取两张图片的 SIFT 特征
image1_sift_features = extract_sift_features('006A1.png')
image2_sift_features = extract_sift_features('006A.png')

# 确保两个 SIFT 特征向量长度一致，用 0 填充较短的向量
max_len = max(len(image1_sift_features), len(image2_sift_features))
image1_sift_features = np.pad(image1_sift_features, (0, max_len - len(image1_sift_features)), 'constant')
image2_sift_features = np.pad(image2_sift_features, (0, max_len - len(image2_sift_features)), 'constant')

# 融合 DINOv2 和 SIFT 特征
image1_combined_features = np.concatenate((image1_dinov2_features, image1_sift_features))
image2_combined_features = np.concatenate((image2_dinov2_features, image2_sift_features))

# 计算融合后特征向量的欧氏距离
distance_score = calculate_euclidean_distance(image1_combined_features, image2_combined_features)
print(f"两张图片特征向量的欧氏距离为: {distance_score}")

# 判断是否改装，这里假设距离大于某个值就认为可能存在改装
threshold = 1.5
if distance_score > threshold:
    print("参照 006A.png，006A1.png 对应的轮毂可能存在改装")
else:
    print("参照 006A.png，006A1.png 对应的轮毂不太可能存在改装")