import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置为支持中文的字体，如黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import cv2
import numpy as np

image_pairs = [
    ('007A.png', '007A1.png'),
    ('007A.png', '007A.png'),
    ('006A.png', '006A1.png'),
    ('004A.png', '004A1.png'),
    ('005A.png', '004A1.png'),
    ('006A.png', '005A.png'),
    ('005A.png', '005B.png'),
    ('006A.png', '004A1.png'),
    ('004A.png', '005A.png'),
    ('005A.png', '006A.png')
]


def preprocess_image(image_path):
    # 读取图像并转换为灰度图
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 噪声处理：使用中值滤波去除椒盐噪声
    image = cv2.medianBlur(image, 3)
    # 噪声处理：使用高斯滤波平滑图像
    image = cv2.GaussianBlur(image, (5, 5), 0)

    # 自适应阈值处理
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 5)

    # 多尺度形态学操作
    mask = np.zeros_like(thresh)
    kernel_sizes = [3, 5, 7]
    for kernel_size in kernel_sizes:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        temp_mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        mask = cv2.bitwise_or(mask, temp_mask)
    cv2.imshow('mask', mask)
    return mask


def shape_similarity(mask1, mask2):
    # 查找轮廓
    contours1, _ = cv2.findContours(mask1, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(mask2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 选择最大的轮廓
    if len(contours1) == 0 or len(contours2) == 0:
        return 0
    contour1 = max(contours1, key=cv2.contourArea)
    contour2 = max(contours2, key=cv2.contourArea)

    # 使用 cv2.matchShapes 计算形状相似度
    similarity = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I3, 0)
    return similarity


for pair in image_pairs:
    img_path1, img_path2 = pair
    # 预处理图像并获取掩码
    mask1 = preprocess_image(img_path1)
    mask2 = preprocess_image(img_path2)

    # 调整掩码尺寸为相同大小
    height = max(mask1.shape[0], mask2.shape[0])
    width = max(mask1.shape[1], mask2.shape[1])
    mask1 = cv2.resize(mask1, (width, height))
    mask2 = cv2.resize(mask2, (width, height))

    # 计算形状相似度
    similarity = shape_similarity(mask1, mask2)

    # 可视化结果
    plt.figure(figsize=(15, 5))

    plt.subplot(131)
    plt.imshow(mask1, cmap='gray')
    plt.title(f'{img_path1} 的掩码')

    plt.subplot(132)
    plt.imshow(mask2, cmap='gray')
    plt.title(f'{img_path2} 的掩码')

    plt.subplot(133)
    overlay = cv2.addWeighted(mask1, 0.5, mask2, 0.5, 0)
    plt.imshow(overlay, cmap='jet')
    plt.title(f'重叠区域 (相似度: {similarity:.4f})')

    plt.tight_layout()
    plt.show()

    print(f"图像对 ({img_path1}, {img_path2}) 的形状相似度: {similarity:.4f}")