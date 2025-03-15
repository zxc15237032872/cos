import torch
import cv2
import numpy as np
from transformers import AutoImageProcessor, AutoModel
from huggingface_hub import hf_hub_download
import os
from PIL import Image


import cv2
import numpy as np


def detect_rotation_angle(image1, image2):
    """
    使用ORB特征检测器和特征点匹配来估算两个图像之间的旋转角度差异
    """
    # 初始化ORB特征检测器
    orb = cv2.ORB_create()

    # 检测和计算特征点和描述符
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)

    # 使用Brute-Force匹配器进行特征点匹配
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    # 按距离对匹配进行排序
    matches = sorted(matches, key=lambda x: x.distance)

    # 选择前一定数量的好匹配
    good_matches = matches[:int(0.1 * len(matches))]

    # 提取好匹配的特征点坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 计算单应性矩阵
    H, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

    if H is not None:
        # 从单应性矩阵中提取旋转角度（假设是平面旋转）
        cos_theta = H[0, 0]
        sin_theta = H[0, 1]
        angle = np.arctan2(sin_theta, cos_theta) * 180 / np.pi
        return angle
    else:
        return 0


import cv2
import numpy as np


def rotate_image(image, angle):
    """
    旋转图像，确保旋转后图像完整显示且中心位置不变
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)

    # 计算旋转后的图像边界
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))

    # 调整旋转矩阵以考虑图像的平移
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]

    # 进行旋转操作
    rotated = cv2.warpAffine(image, M, (new_w, new_h))
    return rotated


def extract_dinov2_features(image):
    cache_dir = "D:\\soft\\huggingface\\transformers"
    if not os.path.exists(cache_dir):
        os.makedirs(cache_dir)

    config_file = hf_hub_download(repo_id="facebook/dinov2-base", filename="config.json", cache_dir=cache_dir)
    weights_file = hf_hub_download(repo_id="facebook/dinov2-base", filename="pytorch_model.bin", cache_dir=cache_dir)

    image_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base", cache_dir=cache_dir)
    model = AutoModel.from_pretrained("facebook/dinov2-base", cache_dir=cache_dir)

    img = image_processor(images=image, return_tensors='pt')

    with torch.no_grad():
        outputs = model(**img)
    embedding = outputs.last_hidden_state[:, 0, :].squeeze(1).squeeze(0)
    return embedding.numpy()


def calculate_cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    similarity = dot_product / (norm_vec1 * norm_vec2)
    return similarity


# 读取图像
image1 = cv2.imread('004A.png')
image2 = cv2.imread('004A1.png')

# 检测并校正旋转角度
angle1 = detect_rotation_angle(image1, image2)
rotated_image1 = rotate_image(image1, angle1)

angle2 = detect_rotation_angle(image2)
rotated_image2 = rotate_image(image2, angle2)

# 将OpenCV的BGR图像转换为PIL的RGB图像
pil_image1 = Image.fromarray(cv2.cvtColor(rotated_image1, cv2.COLOR_BGR2RGB))
pil_image2 = Image.fromarray(cv2.cvtColor(rotated_image2, cv2.COLOR_BGR2RGB))

# 提取特征向量
image1_features = extract_dinov2_features(pil_image1)
image2_features = extract_dinov2_features(pil_image2)

# # 计算余弦相似度
# similarity_score = calculate_cosine_similarity(image1_features, image2_features)
# print(f"两张图片的余弦相似度为: {similarity_score}")

def calculate_pearson_correlation(vec1, vec2):
    """计算皮尔逊相关系数"""
    return np.corrcoef(vec1, vec2)[0, 1]

# 计算皮尔逊相关系数
similarity_score = calculate_pearson_correlation(image1_features, image2_features)
print(f"两张图片特征向量的皮尔逊相关系数为: {similarity_score}")

# 判断是否相似，这里假设相关系数绝对值小于某个值就认为可能存在差异，具体阈值需根据实际情况调整
threshold = 0.8
if abs(similarity_score) < threshold:
    print("两张轮毂图片可能存在差异")
else:
    print("两张轮毂图片不太可能存在差异")

# 判断是否相似，这里假设相似度低于0.8就认为可能存在差异
threshold = 0.8
if similarity_score < threshold:
    print("两张轮毂图片可能存在差异")
else:
    print("两张轮毂图片不太可能存在差异")