import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取图像并进行预处理
def preprocess_image(image_path):
    # 读取图像
    image = cv2.imread(image_path)
    # 灰度化
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Canny 边缘检测
    edges = cv2.Canny(blurred, 50, 150)
    return image, edges

# 霍夫变换检测直线
def detect_lines(edges):
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)
    return lines

# 筛选从中心辐射出去的直线
def filter_lines(lines, center):
    filtered_lines = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            # 计算直线中点
            mid_x = (x1 + x2) // 2
            mid_y = (y1 + y2) // 2
            # 计算中点到中心的距离
            distance = np.sqrt((mid_x - center[0]) ** 2 + (mid_y - center[1]) ** 2)
            if distance < 10:  # 可以根据实际情况调整该阈值
                filtered_lines.append(line)
    return filtered_lines

# 计算辐条角度并补全辐条
def calculate_and_complete_spokes(filtered_lines, center, image_shape):
    spoke_angles = []
    if filtered_lines:
        for line in filtered_lines:
            x1, y1, x2, y2 = line[0]
            # 计算直线与水平方向的夹角
            angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
            spoke_angles.append(angle)

    # 对角度进行排序
    spoke_angles.sort()

    # 计算平均角度间隔
    if len(spoke_angles) > 1:
        angle_differences = np.diff(spoke_angles)
        average_angle_difference = np.mean(angle_differences)
    else:
        average_angle_difference = 360

    # 补全缺失的辐条
    complete_spoke_angles = spoke_angles.copy()
    expected_angle = spoke_angles[0]
    while expected_angle < spoke_angles[-1] + average_angle_difference:
        if not any(np.abs(np.array(spoke_angles) - expected_angle) < average_angle_difference / 2):
            complete_spoke_angles.append(expected_angle)
        expected_angle += average_angle_difference

    # 计算补全后的辐条数量
    spoke_count = len(complete_spoke_angles)

    # 绘制补全后的辐条
    result_image = np.copy(image)
    radius = min(image_shape[0], image_shape[1]) // 2
    for angle in complete_spoke_angles:
        x = int(center[0] + radius * np.cos(np.radians(angle)))
        y = int(center[1] + radius * np.sin(np.radians(angle)))
        cv2.line(result_image, center, (x, y), (0, 255, 0), 2)

    return spoke_count, result_image

# 主函数
if __name__ == "__main__":
    image_path = '005A.png'  # 替换为实际图像路径
    image, edges = preprocess_image(image_path)

    # 计算图像中心
    center = (image.shape[1] // 2, image.shape[0] // 2)

    # 检测直线
    lines = detect_lines(edges)

    # 筛选直线
    filtered_lines = filter_lines(lines, center)

    # 计算并补全辐条
    spoke_count, result_image = calculate_and_complete_spokes(filtered_lines, center, image.shape)

    print(f"轮毂的辐条数: {spoke_count}")

    # 显示结果
    plt.figure(figsize=(10, 5))
    plt.subplot(121)
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title('原始图像')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB))
    plt.title('检测并补全辐条后的图像')
    plt.axis('off')

    plt.show()