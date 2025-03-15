import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import vit_b_16
import matplotlib.pyplot as plt
import requests


# 数据预处理函数
def preprocess_image(image):
    blurred = cv2.GaussianBlur(image, (5, 5), 0)
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)
    equalized = cv2.equalizeHist(gray)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(equalized, -1, kernel)
    return cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)


# 下载微调后的模型
def download_fine_tuned_model(url, save_path='fine_tuned_vit.pth'):
    response = requests.get(url, stream=True)
    if response.status_code == 200:
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                f.write(chunk)
        print(f"模型已成功下载到 {save_path}")
    else:
        print(f"下载失败，状态码: {response.status_code}")


# 提取特征函数，使用微调后的模型
def extract_deep_features(image):
    # 加载预训练的 ViT 模型结构
    model = vit_b_16()
    # 修改最后一层以适应你的任务
    num_ftrs = model.hidden_dim
    model.heads = nn.Linear(num_ftrs, 2)  # 假设是二分类任务，与微调时保持一致

    # 检测设备
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 下载微调后的模型（如果不存在）
    model_path = 'vit_b_16'
    try:
        torch.load(model_path)
    except FileNotFoundError:
        model_url = "vit_b_16"  # 替换为实际的下载链接
        download_fine_tuned_model(model_url, model_path)

    # 加载微调后的模型权重
    model.load_state_dict(torch.load(model_path, map_location=device))
    # 将模型移动到相应设备
    model = model.to(device)
    model.eval()

    # 图像预处理
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    image = transform(image).unsqueeze(0).to(device)

    # 提取特征
    with torch.no_grad():
        features = model(image)
    return features.squeeze().cpu().numpy()


# 计算特征相似度函数
def match_deep_features(features1, features2):
    distance = np.linalg.norm(features1 - features2)
    similarity = 1 / (1 + distance)
    return similarity


# 注册图像函数
def register_images(image1, image2, keypoints1, keypoints2, good_matches):
    if len(good_matches) < 4:
        print("匹配点数量仍然不足，无法计算单应性矩阵。")
        return None
    # 获取匹配点的坐标
    src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # 计算透视变换矩阵
    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    # 应用透视变换
    h, w = image1.shape[:2]
    registered_image = cv2.warpPerspective(image1, M, (image2.shape[1], image2.shape[0]))
    return registered_image


# 主函数
def main():
    image_path1 = '005A.png'
    image_path2 = '005B.png'
    image1 = cv2.imread(image_path1)
    image2 = cv2.imread(image_path2)

    if image1 is None or image2 is None:
        print("无法读取图像，请检查图像路径。")
        return

    # 图像预处理
    image1 = preprocess_image(image1)
    image2 = preprocess_image(image2)

    # 提取特征
    features1 = extract_deep_features(image1)
    features2 = extract_deep_features(image2)

    # 计算相似度
    similarity = match_deep_features(features1, features2)
    print(f"两张照片的相似度: {similarity * 100:.2f}%")

    # 这里使用 SIFT 来获取关键点和匹配点用于图像配准和可视化
    sift = cv2.SIFT_create()
    keypoints1, descriptors1 = sift.detectAndCompute(image1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(image2, None)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(descriptors1, descriptors2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 注册图像
    registered_image = register_images(image1, image2, keypoints1, keypoints2, good_matches)

    # 使用 matplotlib 展示两张待配准的图片
    fig, axes = plt.subplots(1, 2, figsize=(15, 8))

    # 展示第一张待配准图片
    axes[0].imshow(cv2.cvtColor(image1, cv2.COLOR_BGR2RGB))
    axes[0].set_title('Reference Image')
    axes[0].axis('off')

    # 展示第二张待配准图片
    axes[1].imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))
    axes[1].set_title('Test Image')
    axes[1].axis('off')

    plt.tight_layout()
    plt.show()

    # 使用 matplotlib 展示 Matches 和 Registered Image
    if registered_image is not None:
        fig, axes = plt.subplots(1, 2, figsize=(15, 8))

        # 展示匹配结果
        img_matches = cv2.drawMatches(image1, keypoints1, image2, keypoints2, good_matches, None,
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        axes[0].imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
        axes[0].set_title('Matches')
        axes[0].axis('off')

        # 展示配准后的图像
        axes[1].imshow(cv2.cvtColor(registered_image, cv2.COLOR_BGR2RGB))
        axes[1].set_title('Registered Image')
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()
