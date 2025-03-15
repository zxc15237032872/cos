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


def calculate_spokes_and_sectors(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("无法读取图像，请检查图像路径是否正确。")
        return

    # 对数变换
    c = 255 / np.log(1 + np.max(image))
    log_transformed = c * np.log(1 + image.astype(np.float64))
    log_transformed = np.array(log_transformed, dtype=np.uint8)

    # 转换为灰度图
    gray = cv2.cvtColor(log_transformed, cv2.COLOR_BGR2GRAY)

    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Canny 边缘检测
    edges = cv2.Canny(blurred, 50, 150)

    # 霍夫直线变换
    lines = cv2.HoughLines(edges, 1, np.pi / 180, threshold=100)

    if lines is None:
        print("未检测到直线，请检查图像或调整参数。")
        return

    # 提取直线的角度
    angles = []
    for line in lines:
        rho, theta = line[0]
        angle = theta * 180 / np.pi
        angles.append(angle)

    # 对角度进行聚类，确定辐条的数量
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

    num_sectors = len(spoke_angles)
    spoke_angles.sort()

    # 获取图像的高度和宽度
    height, width = image.shape[:2]
    # 计算图像的中心位置
    center_x, center_y = width // 2, height // 2

    # 计算每个扇区的角度
    sector_angle = 360 / num_sectors

    # 初始化一个列表来存储每个扇区的掩码
    sector_masks = []

    # 遍历每个扇区
    for i in range(num_sectors):
        # 计算当前扇区的起始角度和结束角度
        start_angle = spoke_angles[i]
        end_angle = spoke_angles[(i + 1) % num_sectors]
        if end_angle < start_angle:
            end_angle += 360

        # 创建一个掩码，用于提取当前扇区的像素
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(mask, (center_x, center_y), (width // 2, height // 2), 0, start_angle, end_angle, 255, -1)
        sector_masks.append(mask)

    print(f"扇区个数: {num_sectors}")
    print(f"每个扇区的角度: {sector_angle} 度")

    # 显示原始图像、对数变换后的图像和前几个扇区的掩码
    plt.figure(figsize=(20, 5))
    plt.subplot(1, num_sectors + 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('原始图像')
    plt.axis('off')

    plt.subplot(1, num_sectors + 2, 2)
    plt.imshow(cv2.cvtColor(log_transformed, cv2.COLOR_BGR2RGB))
    plt.title('对数变换后的图像')
    plt.axis('off')

    for i in range(min(num_sectors, 6)):
        plt.subplot(1, num_sectors + 2, i + 3)
        plt.imshow(sector_masks[i], cmap='gray')
        plt.title(f'扇区 {i + 1} 掩码')
        plt.axis('off')

    plt.show()

    return num_sectors, sector_angle, sector_masks


# 调用函数进行计算
image_path = '005A.png'
result = calculate_spokes_and_sectors(image_path)