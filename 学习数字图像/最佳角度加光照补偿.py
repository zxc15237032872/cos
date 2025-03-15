import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from skimage.metrics import structural_similarity as ssim

# 解决中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 显示图像
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
matplotlib.use('TkAgg')


def find_best_rotation_angle(image):
    height, width = image.shape
    center = (width // 2, height // 2)
    max_similarity = 0
    best_angle = 0
    # 尝试不同的旋转角度（从 1 度到 180 度）
    for angle in range(1, 181):
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)

        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        cv2.imshow('rotated_image', rotated_image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        # 计算结构相似性指数
        similarity = ssim(image, rotated_image)
        if similarity > max_similarity:
            max_similarity = similarity
            best_angle = angle
    # 计算完整旋转周期的角度
    full_rotation_angle = 360 // (360 // best_angle)
    return full_rotation_angle


def illumination_compensation(image):
    full_rotation_angle = find_best_rotation_angle(image)
    num_rotations = 360 // full_rotation_angle
    rotated_images = []

    height, width = image.shape
    center = (width // 2, height // 2)

    for i in range(num_rotations):
        angle = full_rotation_angle * i
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
        rotated_images.append(rotated_image)

    # 计算旋转图像的平均值
    average_image = np.mean(rotated_images, axis=0).astype(np.uint8)

    # 计算平均光照强度
    mean_illumination = np.mean(average_image)

    # 进行光照校正
    corrected_image = image * (mean_illumination / average_image)
    corrected_image = np.uint8(np.clip(corrected_image, 0, 255))

    return corrected_image


# 读取图像
image = cv2.imread('006A.png', 0)  # 以灰度模式读取图像

# 调用光照补偿函数
corrected_image = illumination_compensation(image)

# 显示原始图像和校正后的图像
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('原始图像')
plt.axis('off')

plt.subplot(122)
plt.imshow(corrected_image, cmap='gray')
plt.title('光照校正后的图像')
plt.axis('off')

plt.show()