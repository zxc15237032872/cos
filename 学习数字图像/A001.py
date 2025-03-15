import cv2
import numpy as np
import matplotlib.pyplot as plt
# 读取原始图像
import matplotlib
matplotlib.use('TkAgg')
import cv2
import numpy as np


import cv2
import numpy as np

import cv2
import numpy as np


def preprocess_image():
    # 读取图像
    image = cv2.imread('007A.png')
    if image is None:
        raise FileNotFoundError('无法读取图像 005A.png')
    plt.figure(figsize=(15, 2))

    # 缩小图像到 30x30
    small_image = cv2.resize(image, (30, 30), interpolation = cv2.INTER_AREA)
    plt.subplot(1, 10, 1)
    plt.imshow(small_image)
    plt.title('small Image')


    # 放大图像到 640x640
    large_image = cv2.resize(small_image, (640, 640), interpolation = cv2.INTER_CUBIC)
    plt.subplot(1, 10, 2)
    plt.imshow(large_image)
    plt.title('large Image')

    # 转换为灰度图
    gray = cv2.cvtColor(large_image, cv2.COLOR_BGR2GRAY)
    plt.subplot(1, 10, 3)
    plt.imshow(gray, cmap='gray')
    plt.title('Gray Image')

    # 高斯模糊平滑图像，减少噪声影响，高斯核大小(5, 5)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    plt.subplot(1, 10, 4)
    plt.imshow(blurred, cmap='gray')
    plt.title('Blurred Image')



    # 中值滤波进一步去除杂质
    median = cv2.medianBlur(blurred, 3)
    plt.subplot(1, 10, 5)
    plt.imshow(median, cmap='gray')
    plt.title('Median Image')

    # 自适应阈值二值化
    binary = cv2.adaptiveThreshold(median, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 301, 1)
    plt.subplot(1, 10, 6)
    plt.imshow(binary, cmap='gray')
    plt.title('Binary Image')
    # 形态学开运算优化轮廓，5x5的结构元素，进行2次开运算



    # 形态学闭运算优化轮廓，5x5的结构元素，进行2次闭运算
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations = 1)

    # 寻找轮廓
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建空白图像，用于绘制处理后的轮廓
    clean_image = np.zeros_like(large_image)

    # 绘制最大轮廓（假设最大轮廓为轮毂）
    max_contour = max(contours, key = cv2.contourArea)
    cv2.drawContours(clean_image, [max_contour], -1, (255, 255, 255), thickness = cv2.FILLED)
    plt.show()
    return clean_image


result_image = preprocess_image()
cv2.imshow('Processed Image', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()