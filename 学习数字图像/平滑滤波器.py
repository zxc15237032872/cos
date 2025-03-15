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
    log_transformed = image

    # 平滑滤波（高斯平滑）
    smoothed = cv2.GaussianBlur(log_transformed, (5, 5), 0)

    # 转换为灰度图
    gray = cv2.cvtColor(smoothed, cv2.COLOR_BGR2GRAY)

    # 设置亮度阈值为 20%
    threshold_value = int(0.2 * 255)
    _, binary = cv2.threshold(gray, threshold_value, 255, cv2.THRESH_BINARY)

    # 查找轮廓
    contours, _ = cv2.findContours(binary.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("未检测到轮廓，请检查图像或调整参数。")
        return

    # 假设最大的轮廓为轮毂
    hub_contour = max(contours, key=cv2.contourArea)

    # 计算轮毂的中心和半径
    ((x, y), radius) = cv2.minEnclosingCircle(hub_contour)
    center = (int(x), int(y))
    radius = int(radius)

    # 霍夫直线变换
    lines = cv2.HoughLines(binary, 1, np.pi / 180, threshold=50)

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
        mask = np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        cv2.ellipse(mask, center, (radius, radius), 0, start_angle, end_angle, 255, -1)
        sector_masks.append(mask)

    print(f"扇区个数: {num_sectors}")
    sector_angles = [(spoke_angles[(i + 1) % num_sectors] - spoke_angles[i]) % 360 for i in range(num_sectors)]
    print(f"每个扇区的角度: {sector_angles} 度")

    # 显示原始图像、对数变换后的图像、平滑滤波后的图像、二值化图像和前几个扇区的掩码
    plt.figure(figsize=(20, 10))
    plt.subplot(2, num_sectors + 2, 1)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('原始图像')
    plt.axis('off')

    plt.subplot(2, num_sectors + 2, 2)
    plt.imshow(cv2.cvtColor(log_transformed, cv2.COLOR_BGR2RGB))
    plt.title('对数变换后的图像')
    plt.axis('off')

    plt.subplot(2, num_sectors + 2, 3)
    plt.imshow(cv2.cvtColor(smoothed, cv2.COLOR_BGR2RGB))
    plt.title('平滑滤波后的图像')
    plt.axis('off')

    plt.subplot(2, num_sectors + 2, 4)
    plt.imshow(binary, cmap='gray')
    plt.title('二值化图像')
    plt.axis('off')

    for i in range(min(num_sectors, 6)):
        plt.subplot(2, num_sectors + 2, i + 5)
        plt.imshow(sector_masks[i], cmap='gray')
        plt.title(f'扇区 {i + 1} 掩码')
        plt.axis('off')

    plt.show()

    return num_sectors, sector_angles, sector_masks


# 调用函数进行计算
image_path = '005A.png'
result = calculate_spokes_and_sectors(image_path)