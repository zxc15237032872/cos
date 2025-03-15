import numpy as np
from skimage.feature import local_binary_pattern
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import cv2

def convert_to_square(image, size=64):
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

def extract_lbp_features(image, radius=3, n_points=8 * 3):
    all_hist = []
    image = np.array(image)
    for channel in range(image.shape[2]):
        channel_image = image[:, :, channel]
        lbp = local_binary_pattern(channel_image, n_points, radius, method='uniform')
        (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-7)
        all_hist.extend(hist)
    return np.array(all_hist)

def cosine_similarity(feature1, feature2):
    dot_product = np.dot(feature1, feature2)
    norm_feature1 = np.linalg.norm(feature1)
    norm_feature2 = np.linalg.norm(feature2)
    similarity = dot_product / (norm_feature1 * norm_feature2)
    return similarity

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

def find_best_rotation_angle(image_path1, image_path2):
    image1 = Image.open(image_path1).convert('RGB')
    image2 = Image.open(image_path2).convert('RGB')

    # 转换为正圆（内切于正方形）
    square_image1 = convert_to_square(image1)
    square_image2 = convert_to_square(image2)

    base_features1 = extract_lbp_features(square_image1)

    max_similarity = -1
    best_angle = 0
    best_rotated_image2 = None

    # 在 0 到 72 度范围内旋转
    for angle in range(1, 100):
        rotated_image2 = rotate_image_opencv(square_image2, angle)
        features2 = extract_lbp_features(rotated_image2)
        similarity = cosine_similarity(base_features1, features2)

        if similarity > max_similarity:
            max_similarity = similarity
            best_angle = angle
            best_rotated_image2 = rotated_image2

    print(f"图片 {image_path1} 和 {image_path2} 的最佳旋转角度为: {best_angle} 度")

    return square_image1, square_image2, best_rotated_image2, best_angle

# 输入图片路径
image_path1 = '006A.png'
image_path2 = '006A1.png'

# 找到最佳旋转角度并获取处理后的图片
square_image1, square_image2, best_rotated_image2, best_angle = find_best_rotation_angle(image_path1, image_path2)

# 显示图片
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(np.array(square_image1))
axes[0].set_title(image_path1)
axes[1].imshow(np.array(square_image2))
axes[1].set_title(image_path2)
axes[2].imshow(np.array(best_rotated_image2))
axes[2].set_title(f"{image_path2} 旋转 {best_angle} 度后")

for ax in axes:
    ax.axis('off')

plt.show()