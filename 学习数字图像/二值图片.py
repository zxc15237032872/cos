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

def count_spokes_from_binary(binary_image):
    # 查找轮廓
    contours, _ = cv2.findContours(binary_image.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if len(contours) == 0:
        print("未检测到轮廓，请检查图像或调整参数。")
        return

    # 假设最大的轮廓为轮毂
    hub_contour = max(contours, key=cv2.contourArea)

    # 计算轮毂的中心和半径
    ((x, y), radius) = cv2.minEnclosingCircle(hub_contour)
    center = (int(x), int(y))

    radius = int(radius)
    print(f"轮毂中心: {center}, 半径: {radius}")

    # 存储每个半径下检测到的辐条数量
    spoke_counts = []

    # 以一定步长增加半径
    step = 1
    for r in range(1, radius, step):
        spoke_count = 0
        prev_pixel = binary_image[center[1], center[0]]
        for angle in range(0, 360):
            # 计算圆上点的坐标
            x_coord = int(center[0] + r * np.cos(np.radians(angle)))
            y_coord = int(center[1] + r * np.sin(np.radians(angle)))
            # 确保坐标在图像范围内
            if 0 <= x_coord < binary_image.shape[1] and 0 <= y_coord < binary_image.shape[0]:
                current_pixel = binary_image[y_coord, x_coord]
                # 检测像素值变化
                if current_pixel != prev_pixel:
                    spoke_count += 1
                prev_pixel = current_pixel
        spoke_counts.append(spoke_count // 2)  # 因为一次完整的变化（0 -> 255 -> 0）对应一个辐条

    # 统计出现次数最多的辐条数量作为最终结果
    from collections import Counter
    counter = Counter(spoke_counts)
    most_common = counter.most_common(1)[0][0]

    return most_common

# 读取图像并进行预处理（这里假设已经得到二值图像）
image = cv2.imread('005A.png', 0)
# 对数变换
c = 255 / np.log(1 + np.max(image))
log_transformed = c * np.log(1 + image.astype(np.float64))
log_transformed = image
# 平滑滤波（高斯平滑）
smoothed = cv2.GaussianBlur(log_transformed, (5, 5), 0)
# 设置亮度阈值为 20%
threshold_value = int(0.2 * 255)
_, binary = cv2.threshold(smoothed, threshold_value, 255, cv2.THRESH_BINARY)

# 计算辐条数量
spoke_count = count_spokes_from_binary(binary)
print(f"检测到的辐条数量: {spoke_count}")

# 显示二值图像
plt.imshow(binary, cmap='gray')
plt.title('二值图像')
plt.axis('off')
plt.show()