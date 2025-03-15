import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import cv2

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
    print("无法读取图像，请检查图像路径是否正确。")
else:
    # 进行直方图均衡化
    equalized_image = cv2.equalizeHist(image)

    # 计算原始图像和均衡化后图像的直方图
    hist_original = cv2.calcHist([image], [0], None, [256], [0, 256])
    hist_equalized = cv2.calcHist([equalized_image], [0], None, [256], [0, 256])

    # 显示原始图像和均衡化后的图像
    plt.figure(figsize=(12, 6))

    plt.subplot(221)
    plt.imshow(image, cmap='gray')
    plt.title('原始图像')
    plt.axis('off')

    plt.subplot(222)
    plt.imshow(equalized_image, cmap='gray')
    plt.title('直方图均衡化后的图像')
    plt.axis('off')

    plt.subplot(223)
    plt.plot(hist_original)
    plt.title('原始图像直方图')

    plt.subplot(224)
    plt.plot(hist_equalized)
    plt.title('均衡化后图像直方图')

    plt.tight_layout()
    plt.show()

    # 保存均衡化后的图像
    cv2.imwrite('equalized_005A.png', equalized_image)
    print("直方图均衡化后的图像已保存为 equalized_005A.png")