import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import cv2
from sklearn.mixture import GaussianMixture

def convert_to_square(image, size=55):
    # 将图像转换为灰度图以便处理
    gray_image = image.convert('L')
    # 将图像转换为 numpy 数组
    img_array = np.array(gray_image)

    # 找到非零像素的坐标
    rows, cols = np.nonzero(img_array)
    if len(rows) == 0 or len(cols) == 0:
        return Image.new("RGB", (size, size), (0, 0, 0))

    # 计算椭圆的边界
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    # 计算椭圆的长短轴长度
    major_axis = max(max_row - min_row, max_col - min_col)
    minor_axis = min(max_row - min_row, max_col - min_col)

    # 计算缩放比例
    scale_ratio = major_axis / minor_axis if minor_axis != 0 else 1

    # 根据长短轴方向进行缩放
    if max_row - min_row > max_col - min_col:
        new_width = int(image.width * scale_ratio)
        resized_image = image.resize((new_width, image.height))
    else:
        new_height = int(image.height * scale_ratio)
        resized_image = image.resize((image.width, new_height))

    # 重新计算灰度图和非零像素坐标
    gray_resized = resized_image.convert('L')
    resized_array = np.array(gray_resized)
    rows, cols = np.nonzero(resized_array)

    # 再次计算椭圆的边界
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)

    # 裁剪出包含椭圆的正方形区域
    left = min_col
    top = min_row
    right = max_col
    bottom = max_row
    cropped_image = resized_image.crop((left, top, right, bottom))

    # 调整图像大小为指定尺寸
    final_image = cropped_image.resize((size, size))
    return final_image

def rotate_image_opencv(image, angle):
    # 将 PIL 图像转换为 OpenCV 格式
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    # 计算旋转中心
    center = (width // 2, height // 2)
    # 获取旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 进行旋转操作
    rotated_img_array = cv2.warpAffine(img_array, rotation_matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    # 将 OpenCV 图像转换回 PIL 格式
    rotated_image = Image.fromarray(rotated_img_array)
    return rotated_image

def calculate_ssim(image1, image2):
    img1_array = np.array(image1.convert('L'))
    img2_array = np.array(image2.convert('L'))
    return ssim(img1_array, img2_array)

def find_best_rotation_angle(image_path1, image_path2):
    image1 = Image.open(image_path1).convert('RGB')
    image2 = Image.open(image_path2).convert('RGB')

    # 转换为正圆（内切于正方形）
    square_image1 = convert_to_square(image1)
    square_image2 = convert_to_square(image2)

    max_similarity = -1
    best_angle = 0
    best_rotated_image2 = None

    # 细化旋转角度搜索，步长为 0.1 度
    for angle in np.arange(0, 100, 0.1):
        rotated_image2 = rotate_image_opencv(square_image2, angle)
        similarity = calculate_ssim(square_image1, rotated_image2)

        if similarity > max_similarity:
            max_similarity = similarity
            best_angle = angle
            best_rotated_image2 = rotated_image2

    print(f"图片 {image_path1} 和 {image_path2} 的最佳旋转角度为: {best_angle} 度")

    return square_image1, square_image2, best_rotated_image2, best_angle

def color_classification(image):
    img = np.array(image.convert('L'))
    # 设定阈值进行颜色分类
    threshold = np.mean(img)
    colored_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    colored_img[img < threshold] = [0, 255, 0]  # 较黑的设为绿色
    colored_img[img >= threshold] = [255, 0, 0]  # 较白的设为红色
    return Image.fromarray(colored_img)

def anomaly_detection(image):
    img = np.array(image.convert('L')).flatten().reshape(-1, 1)
    # 使用高斯混合模型进行异常检测
    gmm = GaussianMixture(n_components=2)
    gmm.fit(img)
    scores = gmm.score_samples(img)
    threshold = np.percentile(scores, 5)  # 设定异常阈值
    anomalies = scores < threshold
    anomaly_img = np.zeros((image.height, image.width), dtype=np.uint8)
    anomaly_img.flat[anomalies] = 255
    return Image.fromarray(anomaly_img)

def extract_anomaly_features(image):
    img = np.array(image)
    return img.flatten()

def cosine_similarity(feature1, feature2):
    dot_product = np.dot(feature1, feature2)
    norm_feature1 = np.linalg.norm(feature1)
    norm_feature2 = np.linalg.norm(feature2)
    if norm_feature1 == 0 or norm_feature2 == 0:
        return 0
    similarity = dot_product / (norm_feature1 * norm_feature2)
    return similarity

# 输入图片路径
image_path1 = '005A.png'
image_path2 = '005B.png'

# 找到最佳旋转角度并获取处理后的图片
square_image1, square_image2, best_rotated_image2, best_angle = find_best_rotation_angle(image_path1, image_path2)

# 颜色分类
colored_image1 = color_classification(square_image1)
colored_image2 = color_classification(square_image2)
colored_rotated_image2 = color_classification(best_rotated_image2)

# 异常检测
anomaly_image1 = anomaly_detection(colored_image1)
anomaly_image2 = anomaly_detection(colored_image2)
anomaly_rotated_image2 = anomaly_detection(colored_rotated_image2)

# 提取异常特征
feature1 = extract_anomaly_features(anomaly_image1)
feature2 = extract_anomaly_features(anomaly_image2)
feature2_rotated = extract_anomaly_features(anomaly_rotated_image2)

# 计算余弦相似度
similarity_before_rotation = cosine_similarity(feature1, feature2)
similarity_after_rotation = cosine_similarity(feature1, feature2_rotated)

print(f"旋转前 {image_path1} 和 {image_path2} 的相似度: {similarity_before_rotation}")
print(f"旋转 {best_angle} 度后 {image_path1} 和 {image_path2} 的相似度: {similarity_after_rotation}")

# 显示图片
fig, axes = plt.subplots(3, 3, figsize=(15, 15))
axes[0, 0].imshow(np.array(square_image1))
axes[0, 0].set_title(image_path1)
axes[0, 1].imshow(np.array(square_image2))
axes[0, 1].set_title(image_path2)
axes[0, 2].imshow(np.array(best_rotated_image2))
axes[0, 2].set_title(f"{image_path2} 旋转 {best_angle} 度后")

axes[1, 0].imshow(np.array(colored_image1))
axes[1, 0].set_title(f"{image_path1} 颜色分类后")
axes[1, 1].imshow(np.array(colored_image2))
axes[1, 1].set_title(f"{image_path2} 颜色分类后")
axes[1, 2].imshow(np.array(colored_rotated_image2))
axes[1, 2].set_title(f"{image_path2} 旋转 {best_angle} 度颜色分类后")

axes[2, 0].imshow(np.array(anomaly_image1), cmap='gray')
axes[2, 0].set_title(f"{image_path1} 异常检测后")
axes[2, 1].imshow(np.array(anomaly_image2), cmap='gray')
axes[2, 1].set_title(f"{image_path2} 异常检测后")
axes[2, 2].imshow(np.array(anomaly_rotated_image2), cmap='gray')
axes[2, 2].set_title(f"{image_path2} 旋转 {best_angle} 度异常检测后")

for ax in axes.flat:
    ax.axis('off')

plt.show()