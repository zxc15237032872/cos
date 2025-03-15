import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import normalize
import torch
from transformers import AutoImageProcessor, AutoModel
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


def preprocess_image(img):
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 自适应直方图均衡化
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    # 边缘增强（使用拉普拉斯算子）
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    laplacian = cv2.convertScaleAbs(laplacian)
    return laplacian


def extract_lbp_features(img):
    # LBP特征提取
    lbp = local_binary_pattern(img, 8, 1, method='uniform')
    hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 10))
    hist = normalize(hist.reshape(1, -1), norm='l2').flatten()
    return hist


def compare_similarity(img1, img2):
    img1_processed = preprocess_image(img1)
    img2_processed = preprocess_image(img2)

    lbp_hist1 = extract_lbp_features(img1_processed)
    lbp_hist2 = extract_lbp_features(img2_processed)

    # 计算LBP特征的相似度
    lbp_similarity = 1 - np.linalg.norm(lbp_hist1 - lbp_hist2)
    print(f"LBP相似度: {lbp_similarity * 100:.2f}%")

    # 计算SSIM相似度
    ssim_similarity = ssim(img1_processed, img2_processed)
    print(f"SSIM相似度: {ssim_similarity * 100:.2f}%")

    # 综合相似度：可以根据实际情况调整权重
    combined_similarity = 0.7 * lbp_similarity + 0.3 * ssim_similarity

    return combined_similarity


def rotate_image(img, angle):
    height, width = img.shape[:2]
    center = (width // 2, height // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(img, M, (width, height))

    return rotated_img


if __name__ == "__main__":
    img1_path = '006A1.png'
    img2_path = '006A.png'

    img1 = cv2.imread(img1_path)
    img2 = cv2.imread(img2_path)
    img1 = cv2.resize(img1, (640, 640))
    img2 = cv2.resize(img2, (640, 640))

    if img1 is None or img2 is None:
        print("无法读取图像，请检查图像路径和文件名。")
    else:
        best_angle = 0
        best_similarity = 0
        best_rotated_img2 = None

        # 从0度到60度寻找相似度最高的角度（使用LBP和SSIM）
        for angle in range(0, 61):
            rotated_img2 = rotate_image(img2, angle)
            similarity = compare_similarity(img1, rotated_img2)

            if similarity > best_similarity:
                best_similarity = similarity
                best_angle = angle
                best_rotated_img2 = rotated_img2

        print(f"使用LBP和SSIM找到的相似度最高的角度: {best_angle} 度")
        print(f"最高相似度: {best_similarity * 100:.2f}%")

        # 使用DINOv2特征计算最佳旋转角度下的相似度
        image1_dinov2_features = extract_dinov2_features(img1_path)
        best_rotated_img2_path = 'temp_best_rotated_img2.png'
        cv2.imwrite(best_rotated_img2_path, best_rotated_img2)
        image2_dinov2_features = extract_dinov2_features(best_rotated_img2_path)

        distance_score = calculate_euclidean_distance(image1_dinov2_features, image2_dinov2_features)
        print(f"使用DINOv2特征计算的欧氏距离: {distance_score}")

        # 判断是否改装，这里假设距离大于某个值就认为可能存在改装
        threshold = 1.5
        if distance_score > threshold:
            print("参照 006A.png，006A1.png 对应的轮毂可能存在改装")
        else:
            print("参照 006A.png，006A1.png 对应的轮毂不太可能存在改装")

        os.remove(best_rotated_img2_path)  # 删除临时保存的图片