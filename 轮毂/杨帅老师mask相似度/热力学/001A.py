import torch
from transformers import AutoImageProcessor, AutoModel
import numpy as np
from huggingface_hub import hf_hub_download
import os
from PIL import Image


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


def calculate_euclidean_distance(vec1, vec2):
    distance = np.linalg.norm(vec1 - vec2)
    return distance


# 提取两张图片的特征向量
image1_features = extract_dinov2_features('006A1.png')
image2_features = extract_dinov2_features('006A.png')

# 计算欧氏距离
distance_score = calculate_euclidean_distance(image1_features, image2_features)
print(f"两张图片特征向量的欧氏距离为: {distance_score}")

# 判断是否改装，这里假设距离大于某个值就认为可能存在改装
threshold = 1.5
if distance_score > threshold:
    print("参照004A.png，005A.png对应的轮毂可能存在改装")
else:
    print("参照004A.png，005A.png对应的轮毂不太可能存在改装")