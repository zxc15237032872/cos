import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置为支持中文的字体，如黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

def register_images(image_path1, image_path2):
    # 读取图像
    img1 = cv2.imread(image_path1, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(image_path2, cv2.IMREAD_GRAYSCALE)

    # 初始化SIFT检测器
    sift = cv2.SIFT_create(nfeatures=2000, contrastThreshold=0.01, edgeThreshold=10)

    # 检测关键点和计算描述子
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 使用FLANN匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 应用比例测试
    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # 提取匹配点
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 计算单应性矩阵
    if len(good_matches) > 4:
        H, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    else:
        print("匹配的特征点不足，无法计算单应性矩阵")
        return

    # 对图像2进行变换
    h, w = img1.shape
    img2_aligned = cv2.warpPerspective(img2, H, (w, h))

    # 绘制结果
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.imshow(img1, cmap='gray')
    plt.title('车A (参照图像)')
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(img2, cmap='gray')
    plt.title('车B (原始图像)')
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(img2_aligned, cmap='gray')
    plt.title('车B (配准后)')
    plt.axis('off')

    plt.show()

# 输入图像路径
image_path1 = 'che001.png'
image_path2 = 'che002.png'

# 调用配准函数
register_images(image_path1, image_path2)