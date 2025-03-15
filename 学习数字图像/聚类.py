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
# 调整图像大小
image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_AREA)

# 检查图像是否成功读取
if image is None:
    print("无法读取图像，请检查图像路径和文件名是否正确。")
else:
    # 定义结构元素
    kernel = np.ones((4, 4), np.uint8)

    # 进行腐蚀操作
    eroded_image = cv2.erode(image, kernel, iterations=1)

    # 光照补偿部分
    # 获取图像的高度和宽度
    height, width = eroded_image.shape[:2]
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
                angle_avg[angle] += eroded_image[y, x]

    # 计算每个角度的平均灰度值
    angle_avg /= min(center_x, center_y) - 1

    # 生成光照补偿图
    compensation_map = np.zeros_like(eroded_image)
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
    compensated_image = eroded_image * (mean_illumination / compensation_map)
    compensated_image = np.uint8(np.clip(compensated_image, 0, 255))

    # 进行膨胀操作
    final_image = cv2.dilate(compensated_image, kernel, iterations=1)

    # 图像聚类生成轮毂掩码
    # 将图像转换为一维数组
    pixels = final_image.reshape((-1, 1)).astype(np.float32)

    # 定义 K - 均值聚类的参数
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2  # 聚类的类别数
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 假设轮毂对应的类别是灰度值较大的类别
    hub_class = np.argmax(centers)
    mask = (labels == hub_class).reshape(final_image.shape).astype(np.uint8) * 255

    # 显示原始图像、腐蚀后的图像、光照补偿后的图像、最终图像和轮毂掩码
    plt.figure(figsize=(25, 5))

    plt.subplot(151)
    plt.imshow(image, cmap='gray')
    plt.title('原始图像')
    plt.axis('off')

    plt.subplot(152)
    plt.imshow(eroded_image, cmap='gray')
    plt.title('腐蚀后的图像')
    plt.axis('off')

    plt.subplot(153)
    plt.imshow(compensated_image, cmap='gray')
    plt.title('光照补偿后的图像')
    plt.axis('off')

    plt.subplot(154)
    plt.imshow(final_image, cmap='gray')
    plt.title('最终处理后的图像')
    plt.axis('off')

    plt.subplot(155)
    plt.imshow(mask, cmap='gray')
    plt.title('轮毂掩码')
    plt.axis('off')

    plt.show()

    # 保存光照补偿后的图像
    cv2.imwrite('005A_compensated.png', compensated_image)
    print("光照补偿后的图像已保存为 005A_compensated.png")

    # 保存最终处理后的图像
    cv2.imwrite('005A_final.png', final_image)
    print("最终处理后的图像已保存为 005A_final.png")

    # 保存轮毂掩码
    cv2.imwrite('005A_mask.png', mask)
    print("轮毂掩码已保存为 005A_mask.png")