
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置为支持中文的字体，如黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import cv2
from sklearn.cluster import KMeans

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
    for angle in np.arange(0, 100, 1):
        rotated_image2 = rotate_image_opencv(square_image2, angle)
        similarity = calculate_ssim(square_image1, rotated_image2)

        if similarity > max_similarity:
            max_similarity = similarity
            best_angle = angle
            best_rotated_image2 = rotated_image2

    print(f"图片 {image_path1} 和 {image_path2} 的最佳旋转角度为: {best_angle} 度")

    return square_image1, square_image2, best_rotated_image2, best_angle

def find_two_main_colors(image):
    """
    使用 K-Means 聚类找到图像中的两种主要颜色
    """
    img = np.array(image)
    pixels = img.reshape(-1, 3)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(pixels)
    labels = kmeans.labels_
    center1, center2 = kmeans.cluster_centers_

    # 确定较暗和较亮的颜色
    if np.mean(center1) < np.mean(center2):
        dark_color, light_color = center1, center2
    else:
        dark_color, light_color = center2, center1

    new_image = np.zeros_like(img)
    for i, label in enumerate(labels):
        if label == 0:
            new_image[i // img.shape[1], i % img.shape[1]] = [0, 0, 0]  # 较暗的设为黑色
        else:
            new_image[i // img.shape[1], i % img.shape[1]] = [255, 0, 0]  # 较亮的设为红色

    return Image.fromarray(new_image.astype(np.uint8))

def extract_features(image):
    img = np.array(image).flatten()
    return img

# def similarity(feature1, feature2):
#     dot_product = np.dot(feature1, feature2)
#     norm_feature1 = np.linalg.norm(feature1)
#     norm_feature2 = np.linalg.norm(feature2)
#     if norm_feature1 == 0 or norm_feature2 == 0:
#         return 0
#     similarity = dot_product / (norm_feature1 * norm_feature2)
#     return similarity
# def similarity(feature1, feature2):
#     distance = np.sum(np.abs(feature1 - feature2))
#     # 将距离转换为相似度，可根据实际情况调整分母
#     similarity = 1 / (1 + distance)
#     return similarity

import numpy as np
from PIL import Image





def similarity(feature1, feature2):
    """
    根据汉明距离计算相似度
    """
    distance = np.sum(np.abs(feature1 - feature2))
    print(distance)
    # 将距离转换为相似度，可根据实际情况调整分母
    similarity =1  / (1+ distance)
    return similarity



# 输入图片路径
image_path1 = '006A.png'
image_path2 = '006A1.png'

# 找到最佳旋转角度并获取处理后的图片
square_image1, square_image2, best_rotated_image2, best_angle = find_best_rotation_angle(image_path1, image_path2)

# 找到两种主要颜色并赋予黑色和红色
colored_image1 = find_two_main_colors(square_image1)
colored_image2 = find_two_main_colors(square_image2)
colored_rotated_image2 = find_two_main_colors(best_rotated_image2)

# 提取特征
feature1 = extract_features(colored_image1)
feature2 = extract_features(colored_image2)
feature2_rotated = extract_features(colored_rotated_image2)

# 计算余弦相似度
similarity_before_rotation = similarity(feature1, feature2)
similarity_after_rotation = similarity(feature1, feature2_rotated)

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
axes[1, 0].set_title(f"{image_path1} 颜色处理后")
axes[1, 1].imshow(np.array(colored_image2))
axes[1, 1].set_title(f"{image_path2} 颜色处理后")
axes[1, 2].imshow(np.array(colored_rotated_image2))
axes[1, 2].set_title(f"{image_path2} 旋转 {best_angle} 度颜色处理后")

axes[2, 0].imshow(np.array(Image.fromarray(feature1.reshape(colored_image1.size[1], colored_image1.size[0], 3))), cmap='gray')
axes[2, 0].set_title(f"{image_path1} 特征可视化")
axes[2, 1].imshow(np.array(Image.fromarray(feature2.reshape(colored_image2.size[1], colored_image2.size[0], 3))), cmap='gray')
axes[2, 1].set_title(f"{image_path2} 特征可视化")
axes[2, 2].imshow(np.array(Image.fromarray(feature2_rotated.reshape(colored_rotated_image2.size[1], colored_rotated_image2.size[0], 3))), cmap='gray')
axes[2, 2].set_title(f"{image_path2} 旋转 {best_angle} 度特征可视化")

for ax in axes.flat:
    ax.axis('off')

plt.show()