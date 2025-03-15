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
    embedding = outputs.last_hidden_state[:, 0, :].squeeze(1).squeeze(0)  # 再添加一个squeeze(0)去除多余维度
    return embedding.numpy()


def calculate_cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity


# 提取两张图片的特征向量
image1_features = extract_dinov2_features('005A.png')
image2_features = extract_dinov2_features('004A.png')

# 计算余弦相似度
similarity_score = calculate_cosine_similarity(image1_features, image2_features)
print(f"两张图片的余弦相似度为: {similarity_score}")

# 判断是否改装，这里假设相似度低于0.8就认为可能存在改装
threshold = 0.8
if similarity_score < threshold:
    print("参照002.png，001.png对应的轮毂可能存在改装")
else:
    print("参照002.png，001.png对应的轮毂不太可能存在改装")