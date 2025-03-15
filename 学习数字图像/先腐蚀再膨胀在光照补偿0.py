import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import find_peaks

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

    # 光照补偿：使用自适应直方图均衡化（CLAHE）
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    compensated_image = clahe.apply(opened_image)

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
    plt.imshow(compensated_image, cmap='gray')
    plt.title('光照补偿后的图像')
    plt.axis('off')

    plt.show()

    # 保存开运算后的图像
    cv2.imwrite('005A_opened.png', opened_image)
    print("开运算后的图像已保存为 005A_opened.png")

    # 保存光照补偿后的图像
    cv2.imwrite('005A_compensated.png', compensated_image)
    print("光照补偿后的图像已保存为 005A_compensated.png")