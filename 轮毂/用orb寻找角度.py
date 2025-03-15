import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from scipy.fftpack import fft2, ifft2, fftshift, ifftshift
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 将图像转换为正方形（内切圆）
def convert_to_square(image, size=64):
    gray_image = image.convert('L')
    img_array = np.array(gray_image)

    rows, cols = np.nonzero(img_array)
    if len(rows) == 0 or len(cols) == 0:
        return Image.new("RGB", (size, size), (0, 0, 0))

    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    major_axis = max(max_row - min_row, max_col - min_col)
    minor_axis = min(max_row - min_row, max_col - min_col)

    scale_ratio = major_axis / minor_axis if minor_axis != 0 else 1

    if max_row - min_row > max_col - min_col:
        new_width = int(image.width * scale_ratio)
        resized_image = image.resize((new_width, image.height))
    else:
        new_height = int(image.height * scale_ratio)
        resized_image = image.resize((image.width, new_height))

    gray_resized = resized_image.convert('L')
    resized_array = np.array(gray_resized)
    rows, cols = np.nonzero(resized_array)

    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)

    left = min_col
    top = min_row
    right = max_col
    bottom = max_row
    cropped_image = resized_image.crop((left, top, right, bottom))

    final_image = cropped_image.resize((size, size))
    return final_image

# 使用傅里叶变换旋转图像
def rotate_image_fourier(image, angle):
    gray_image = image.convert('L')
    img_array = np.array(gray_image)

    # 零填充
    pad_size = max(img_array.shape)
    padded_img = np.pad(img_array, ((0, pad_size - img_array.shape[0]), (0, pad_size - img_array.shape[1])), mode='constant', constant_values=0)

    # 傅里叶变换并将低频分量移到中心
    f = fftshift(fft2(padded_img))

    # 计算旋转矩阵
    theta = np.radians(angle)
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # 频率域旋转
    height, width = padded_img.shape
    rotated_f = np.zeros_like(f, dtype=complex)
    center_y = height // 2
    center_x = width // 2
    for y in range(height):
        for x in range(width):
            new_x = int((x - center_x) * cos_theta - (y - center_y) * sin_theta) + center_x
            new_y = int((x - center_x) * sin_theta + (y - center_y) * cos_theta) + center_y
            if 0 <= new_x < width and 0 <= new_y < height:
                rotated_f[y, x] = f[new_y, new_x]

    # 将低频分量移回原来的位置并进行逆傅里叶变换
    rotated_img_array = np.real(ifft2(ifftshift(rotated_f)))

    # 裁剪回原始大小
    rotated_img_array = rotated_img_array[:img_array.shape[0], :img_array.shape[1]]

    # 将结果转换为 PIL 图像
    rotated_image = Image.fromarray(np.uint8(rotated_img_array))
    return rotated_image

# 提取 ORB 特征
def extract_orb_features(image):
    image = np.array(image)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)
    if descriptors is not None:
        return descriptors.flatten()
    else:
        return np.array([])

# 计算余弦相似度
def cosine_similarity(feature1, feature2):
    if len(feature1) == 0 or len(feature2) == 0:
        return 0
    dot_product = np.dot(feature1, feature2)
    norm_feature1 = np.linalg.norm(feature1)
    norm_feature2 = np.linalg.norm(feature2)
    similarity = dot_product / (norm_feature1 * norm_feature2)
    return similarity

# 找到最佳旋转角度
def find_best_rotation_angle(image_path1, image_path2):
    image1 = Image.open(image_path1).convert('RGB')
    image2 = Image.open(image_path2).convert('RGB')

    square_image1 = convert_to_square(image1)
    square_image2 = convert_to_square(image2)

    base_features1 = extract_orb_features(square_image1)

    max_similarity = -1
    best_angle = 0
    best_rotated_image2 = None

    for angle in range(0, 73):
        rotated_image2 = rotate_image_fourier(square_image2, angle)
        rotated_image2 = rotated_image2.convert('RGB')
        features2 = extract_orb_features(rotated_image2)
        similarity = cosine_similarity(base_features1, features2)

        if similarity > max_similarity:
            max_similarity = similarity
            best_angle = angle
            best_rotated_image2 = rotated_image2

    print(f"图片 {image_path1} 和 {image_path2} 的最佳旋转角度为: {best_angle} 度")

    return square_image1, square_image2, best_rotated_image2, best_angle

# 主程序
if __name__ == "__main__":
    image_path1 = '004A.png'
    image_path2 = '004A1.png'

    square_image1, square_image2, best_rotated_image2, best_angle = find_best_rotation_angle(image_path1, image_path2)

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