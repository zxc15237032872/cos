import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib

# 解决中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 显示图像
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
matplotlib.use('TkAgg')


def detect_spokes(image):
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 边缘检测
    edges = cv2.Canny(blurred, 50, 150)

    # 霍夫变换检测直线
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    if lines is None:
        return []

    # 提取直线的角度
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = theta * 180 / np.pi
        angles.append(angle)

    # 对角度进行聚类，确定辐条的数量和位置
    spoke_angles = []
    angle_threshold = 10
    for angle in angles:
        found = False
        for spoke_angle in spoke_angles:
            if abs(angle - spoke_angle) < angle_threshold:
                found = True
                break
        if not found:
            spoke_angles.append(angle)

    spoke_angles.sort()
    return spoke_angles


def equalize_sectors(image, spoke_angles):
    height, width = image.shape[:2]
    center_x, center_y = width // 2, height // 2
    num_sectors = len(spoke_angles)

    sectors = []
    for i in range(num_sectors):
        start_angle = spoke_angles[i]
        end_angle = spoke_angles[(i + 1) % num_sectors]
        if end_angle < start_angle:
            end_angle += 360

        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(mask, (center_x, center_y), (width // 2, height // 2), 0, start_angle, end_angle, 255, -1)
        sector = cv2.bitwise_and(image, image, mask=mask)
        sectors.append(sector)

    reference_sector = sectors[0]
    reference_mean = np.mean(reference_sector[reference_sector > 0])

    equalized_sectors = []
    for sector in sectors:
        sector_mean = np.mean(sector[sector > 0])
        if sector_mean > 0:
            factor = reference_mean / sector_mean
            adjusted_sector = (sector * factor).astype(np.uint8)
            equalized_sectors.append(adjusted_sector)
        else:
            equalized_sectors.append(sector)

    equalized_image = np.zeros((height, width), dtype=np.uint8)
    for sector in equalized_sectors:
        equalized_image = cv2.bitwise_or(equalized_image, sector)

    return equalized_image


# 读取图像
image = cv2.imread('005A.png')

# 检查图像是否成功读取
if image is None:
    print("无法读取图像，请检查图像路径是否正确。")
else:
    # 检测辐条数量和角度
    spoke_angles = detect_spokes(image)
    if len(spoke_angles) == 0:
        print("未检测到辐条，请检查图像或调整参数。")
    else:
        # 转换为灰度图
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # 进行扇区均衡化处理
        equalized_image = equalize_sectors(gray_image, spoke_angles)

        # 显示原始图像和均衡化后的图像
        plt.figure(figsize=(12, 6))
        plt.subplot(121)
        plt.imshow(gray_image, cmap='gray')
        plt.title('原始图像')
        plt.axis('off')
        plt.subplot(122)
        plt.imshow(equalized_image, cmap='gray')
        plt.title('扇区均衡化后的图像')
        plt.axis('off')
        plt.show()

        # 保存均衡化后的图像
        cv2.imwrite('equalized_sectors_005A.png', equalized_image)
        print("扇区均衡化后的图像已保存为 equalized_sectors_005A.png")