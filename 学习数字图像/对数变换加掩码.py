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

    # 查找轮廓
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 筛选轮毂轮廓（假设最大的轮廓为轮毂）
    if len(contours) == 0:
        print("未检测到轮廓，请检查图像或调整参数。")
        return
    hub_contour = max(contours, key=cv2.contourArea)

    # 计算轮毂的中心和半径
    ((x, y), radius) = cv2.minEnclosingCircle(hub_contour)
    center = (int(x), int(y))
    radius = int(radius)

    # 创建一个空白图像用于绘制掩码
    mask = np.zeros(image.shape[:2], dtype=np.uint8)

    # 计算辐条个数和扇区角度
    # 这里简单通过计算轮廓的凸缺陷来估计辐条个数
    hull = cv2.convexHull(hub_contour, returnPoints=False)
    defects = cv2.convexityDefects(hub_contour, hull)

    if defects is not None:
        spoke_count = 0
        sector_angles = []
        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(hub_contour[s][0])
            end = tuple(hub_contour[e][0])
            far = tuple(hub_contour[f][0])
            # 计算凸缺陷的角度
            angle = np.arctan2(far[1] - center[1], far[0] - center[0]) * 180 / np.pi
            sector_angles.append(angle)
            spoke_count += 1

        # 对扇区角度进行排序
        sector_angles.sort()

        # 计算每个扇区的角度
        sector_angles_diff = []
        for i in range(len(sector_angles)):
            if i == len(sector_angles) - 1:
                diff = (sector_angles[0] + 360) - sector_angles[i]
            else:
                diff = sector_angles[i + 1] - sector_angles[i]
            sector_angles_diff.append(diff)

        # 生成每个扇区的掩码
        sector_masks = []
        for i in range(spoke_count):
            start_angle = sector_angles[i]
            end_angle = sector_angles[(i + 1) % spoke_count]
            if end_angle < start_angle:
                end_angle += 360
            sector_mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.ellipse(sector_mask, center, (radius, radius), 0, start_angle, end_angle, 255, -1)
            sector_masks.append(sector_mask)

        print(f"辐条个数: {spoke_count}")
        print(f"每个扇区的角度: {sector_angles_diff}")

        # 显示原始图像、对数变换后的图像、边缘图像和第一个扇区的掩码
        plt.figure(figsize=(20, 5))
        plt.subplot(141)
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title('原始图像')
        plt.axis('off')
        plt.subplot(142)
        plt.imshow(cv2.cvtColor(log_transformed, cv2.COLOR_BGR2RGB))
        plt.title('对数变换后的图像')
        plt.axis('off')
        plt.subplot(143)
        plt.imshow(edges, cmap='gray')
        plt.title('边缘图像')
        plt.axis('off')
        plt.subplot(144)
        plt.imshow(sector_masks[0], cmap='gray')
        plt.title('第一个扇区的掩码')
        plt.axis('off')
        plt.show()

        return spoke_count, sector_angles_diff, sector_masks
    else:
        print("未检测到辐条，请检查图像或调整参数。")
        return

# 调用函数进行计算
image_path = '005A.png'
result = calculate_spokes_and_sectors(image_path)