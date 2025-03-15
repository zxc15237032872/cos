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

# 读取图像
image = cv2.imread('005A.png', 0)  # 以灰度模式读取图像

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

# 二值化处理
_, binary_image = cv2.threshold(corrected_image, 127, 255, cv2.THRESH_BINARY)

# 形态学操作
kernel = np.ones((10, 10), np.uint8)
closed_image = cv2.morphologyEx(binary_image, cv2.MORPH_CLOSE, kernel)
opened_image = cv2.morphologyEx(closed_image, cv2.MORPH_OPEN, kernel)

# 边缘平滑
smoothed_image = cv2.GaussianBlur(opened_image, (9, 9), 0)

# 掩膜操作
final_image = cv2.bitwise_and(corrected_image, corrected_image, mask=smoothed_image)

# 显示原始图像、光照校正后的图像和最终结果
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
plt.imshow(final_image, cmap='gray')
plt.title('最终结果')
plt.axis('off')

plt.show()