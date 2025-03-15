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
image = cv2.imread('005A.png', cv2.IMREAD_GRAYSCALE)

# 检查图像是否成功读取
if image is None:
    print("无法读取图像，请检查图像路径和文件名是否正确。")
else:
    # 定义结构元素
    kernel = np.ones((5, 5), np.uint8)

    # 进行腐蚀操作
    eroded_image = cv2.erode(image, kernel, iterations=1)

    # 进行膨胀操作
    opened_image = cv2.dilate(eroded_image, kernel, iterations=1)

    # 光照补偿部分
    # 获取图像的高度和宽度
    height, width = opened_image.shape[:2]
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
                angle_avg[angle] += opened_image[y, x]

    # 计算每个角度的平均灰度值
    angle_avg /= min(center_x, center_y) - 1

    # 生成光照补偿图
    compensation_map = np.zeros_like(opened_image)
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
    corrected_image = opened_image * (mean_illumination / compensation_map)
    corrected_image = np.uint8(np.clip(corrected_image, 0, 255))

    # 显示原始图像、腐蚀后的图像、开运算后的图像和光照补偿后的图像
    plt.figure(figsize=(20, 5))

    plt.subplot(141)
    plt.imshow(image, cmap='gray')
    plt.title('原始图像')
    plt.axis('off')

    plt.subplot(142)
    plt.imshow(eroded_image, cmap='gray')
    plt.title('腐蚀后的图像')
    plt.axis('off')

    plt.subplot(143)
    plt.imshow(opened_image, cmap='gray')
    plt.title('开运算后的图像')
    plt.axis('off')

    plt.subplot(144)
    plt.imshow(corrected_image, cmap='gray')
    plt.title('光照补偿后的图像')
    plt.axis('off')

    plt.show()

    # 保存开运算后的图像
    cv2.imwrite('005A_opened.png', opened_image)
    print("开运算后的图像已保存为 005A_opened.png")

    # 保存光照补偿后的图像
    cv2.imwrite('005A_compensated.png', corrected_image)
    print("光照补偿后的图像已保存为 005A_compensated.png")