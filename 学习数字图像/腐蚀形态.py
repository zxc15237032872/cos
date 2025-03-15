import cv2
import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use('TkAgg')

# 读取图像并进行腐蚀操作
def preprocess_image(image_path):
    # 读取图像
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 检查图像是否成功读取
    if image is None:
        print("无法读取图像，请检查图像路径和文件名是否正确。")
        return None, None
    # 定义腐蚀操作的结构元素
    kernel = np.ones((2, 2), np.uint8)
    # 进行腐蚀操作
    eroded_image = cv2.erode(image, kernel, iterations=1)
    return image, eroded_image
# 计算辐条数量
def calculate_spoke_count(eroded_image):
    # 进行二维傅里叶变换
    f = np.fft.fft2(eroded_image)
    # 频谱中心化
    fshift = np.fft.fftshift(f)
    # 计算频谱幅度
    magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-10)  # 避免 log(0) 警告

    # 找到频谱中心
    rows, cols = eroded_image.shape
    crow, ccol = rows // 2, cols // 2

    # 动态调整环形区域半径
    radius_inner = int(min(rows, cols) * 0.1)  # 适当减小内半径
    radius_outer = int(min(rows, cols) * 0.6)  # 适当减小外半径

    # 提取频谱的环形区域
    mask = np.zeros((rows, cols), np.uint8)
    y, x = np.ogrid[:rows, :cols]
    mask_area = (x - ccol) ** 2 + (y - crow) ** 2 <= radius_outer ** 2
    mask_area &= (x - ccol) ** 2 + (y - crow) ** 2 >= radius_inner ** 2
    mask[mask_area] = 1
    fshift_masked = fshift * mask
    magnitude_spectrum_masked = 20 * np.log(np.abs(fshift_masked) + 1e-10)  # 避免 log(0) 警告

    # 计算角度方向上的能量分布
    angles = np.linspace(0, 2 * np.pi, 360)
    energy_angles = []
    for angle in angles:
        x = int(ccol + radius_outer * np.cos(angle))
        y = int(crow + radius_outer * np.sin(angle))
        # 边界检查
        if 0 <= y < rows and 0 <= x < cols:
            energy_angles.append(magnitude_spectrum_masked[y, x])
        else:
            energy_angles.append(0)  # 超出边界则赋值为 0

    # 找到能量分布的峰值
    peaks, _ = find_peaks(energy_angles, height=5, distance=20)

    # 计算相邻峰值之间的平均角度间隔
    if len(peaks) > 1:
        peak_angles = [angles[i] for i in peaks]
        angle_differences = np.diff(peak_angles)
        average_angle_difference = np.mean(angle_differences)
        # 根据平均角度间隔推测完整的辐条数量
        total_spoke_count = int(2 * np.pi / average_angle_difference)
    else:
        total_spoke_count = len(peaks)

    return total_spoke_count, magnitude_spectrum_masked
# 主函数
if __name__ == "__main__":
    image_path = '005B.png'  # 替换为实际图像路径
    original_image, eroded_image = preprocess_image(image_path)

    if original_image is not None:
        spoke_count, magnitude_spectrum_masked = calculate_spoke_count(eroded_image)
        print(f"轮毂的辐条数: {spoke_count}")

        # 显示原始图像、腐蚀后的图像和频谱
        plt.figure(figsize=(15, 5))

        plt.subplot(131)
        plt.imshow(original_image, cmap='gray')
        plt.title('原始图像')
        plt.axis('off')

        plt.subplot(132)
        plt.imshow(eroded_image, cmap='gray')
        plt.title('腐蚀后的图像')
        plt.axis('off')

        plt.subplot(133)
        plt.imshow(magnitude_spectrum_masked, cmap='gray')
        plt.title('频谱（环形区域）')
        plt.axis('off')

        plt.show()