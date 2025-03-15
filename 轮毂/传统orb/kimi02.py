import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置为支持中文的字体，如黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义高斯高通滤波器锐化函数
def gaussian_highpass_sharpen(image, sigma=1.0):
    blurred = cv2.GaussianBlur(image, (0, 0), sigma)
    sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)
    return sharpened

# 定义计算相似度的函数
def calculate_similarity(des1, des2):
    if des1 is None or des2 is None or len(des1) == 0 or len(des2) == 0:
        return 0, []  # 如果描述子为空，直接返回相似度为0

    # 确保描述子为 np.float32 类型
    des1 = np.float32(des1)
    des2 = np.float32(des2)

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    try:
        matches = flann.knnMatch(des1, des2, k=2)
    except cv2.error as e:
        print(f"FLANN匹配器错误：{e}")
        return 0, []  # 如果匹配失败，返回相似度为0

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # 改进相似度计算方法
    similarity = len(good_matches) / min(len(des1), len(des2)) * 100
    return similarity, good_matches

# 图像对
image_pairs = [
    ('007A.png', '007A1.png'),
    ('007A.png', '007A.png'),
    ('004A.png', '004A1.png'),
    ('006A.png', '006A1.png'),
    ('005A.png', '004A1.png'),
    ('006A.png', '005A.png'),
    ('005A.png', '005B.png'),
    ('006A.png', '004A1.png'),
    ('004A.png', '005A.png'),
    ('005A.png', '006A.png')
]

# 遍历图像对
for idx, (img1_path, img2_path) in enumerate(image_pairs):
    # 读取图像
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # 检查图像是否读取成功
    if img1 is None or img2 is None:
        print(f"无法读取图像 {img1_path} 或 {img2_path}")
        continue

    # 使用高斯高通滤波器进行锐化
    img1_sharpened = gaussian_highpass_sharpen(img1, sigma=1.0)
    img2_sharpened = gaussian_highpass_sharpen(img2, sigma=1.0)

    # 初始化SIFT检测器并调整参数
    sift = cv2.SIFT_create(
        nfeatures=2000,  # 增加关键点数量
        nOctaveLayers=3,
        contrastThreshold=0.1,  # 降低对比度阈值以增加关键点
        edgeThreshold=20,
        sigma=1.1
    )

    # 检测关键点和计算描述子
    kp1, des1 = sift.detectAndCompute(img1_sharpened, None)
    kp2, des2 = sift.detectAndCompute(img2_sharpened, None)

    # 计算相似度
    similarity, good_matches = calculate_similarity(des1, des2)

    # 判断是否改装
    if img1_path[:3] == img2_path[:3] and img1_path[3] == img2_path[3]:
        modification_status = "未改装"
    else:
        modification_status = "改装"

    # 输出结果
    print(f"{img1_path} 和 {img2_path} 已知{modification_status}，当前相似度：{similarity:.2f}%")
    print(f"{img1_path} 关键点个数：{len(kp1)}")
    print(f"{img2_path} 关键点个数：{len(kp2)}")
    print("-" * 40)

    # 绘制第一对图像的关键点匹配结果
    if idx == 0:  # 只绘制第一对图像的关键点匹配结果
        img_matches = cv2.drawMatches(img1_sharpened, kp1, img2_sharpened, kp2, good_matches[:30], None,
                                      matchColor=(0, 255, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.figure(figsize=(12, 6))
        plt.imshow(img_matches, cmap='gray')
        plt.title(f"关键点匹配结果：{img1_path} 和 {img2_path}")
        plt.show()