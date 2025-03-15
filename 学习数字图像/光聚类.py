import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 解决中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 显示图像
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
matplotlib.use('TkAgg')


def illumination_compensation(image):
    # 获取图像的高度和宽度
    height, width = image.shape[:2]
    # 计算图像的中心位置
    center_x, center_y = width // 2, height // 2

    # 初始化一个数组来存储每个角度的平均灰度值
    angle_bins = 360
    angle_avg = np.zeros(angle_bins)

    # 计算每个角度的平均灰度值
    for r in range(1, min(center_x, center_y)):
        for angle in range(angle_bins):
            x = int(center_x + r * np.cos(np.radians(angle)))
            y = int(center_y + r * np.sin(np.radians(angle)))
            if 0 <= x < width and 0 <= y < height:
                angle_avg[angle] += image[y, x]

    # 计算每个角度的平均灰度值
    angle_avg /= min(center_x, center_y) - 1

    # 生成光照补偿图
    compensation_map = np.zeros_like(image)
    for r in range(height):
        for c in range(width):
            dx = c - center_x
            dy = r - center_y
            angle = np.arctan2(dy, dx) * 180 / np.pi
            if angle < 0:
                angle += 360
            angle_index = int(angle)
            compensation_map[r, c] = angle_avg[angle_index]

    # 计算平均光照强度
    mean_illumination = np.mean(compensation_map)

    # 进行光照校正
    corrected_image = image * (mean_illumination / compensation_map)
    corrected_image = np.uint8(np.clip(corrected_image, 0, 255))

    return corrected_image


def generate_mask(corrected_image, num_clusters=2):
    # 将图像转换为一维数组
    pixels = corrected_image.reshape((-1, 1))
    pixels = np.float32(pixels)

    # 定义 K - Means 算法的终止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # 应用 K - Means 算法
    _, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 将中心值转换为整数
    centers = np.uint8(centers)

    # 根据标签获取每个像素的聚类中心值
    segmented_image = centers[labels.flatten()]

    # 重新调整形状以匹配原始图像
    segmented_image = segmented_image.reshape(corrected_image.shape)

    # 假设轮毂对应的聚类中心值较大，获取 mask
    if centers[0] > centers[1]:
        mask = (segmented_image == centers[0]).astype(np.uint8) * 255
    else:
        mask = (segmented_image == centers[1]).astype(np.uint8) * 255

    return mask


# 读取图像
image = cv2.imread('005A.png', 0)  # 以灰度模式读取图像

# 调用光照补偿函数
corrected_image = illumination_compensation(image)

# 生成 mask
mask = generate_mask(corrected_image)

# 显示原始图像、光照校正后的图像和 mask
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(image, cmap='gray')
plt.title('原始图像')
plt.axis('off')

plt.subplot(132)
plt.imshow(corrected_image, cmap='gray')
plt.title('光照校正后的图像')
plt.axis('off')

plt.subplot(133)
plt.imshow(mask, cmap='gray')
plt.title('轮毂的 mask')
plt.axis('off')

plt.show()