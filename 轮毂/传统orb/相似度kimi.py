import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置为支持中文的字体，如黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 定义高斯高通滤波器锐化函数
# def gaussian_highpass_sharpen(sharpened, sigma=1.0):
#
#     clahe = cv2.createCLAHE(clipLimit=4, tileGridSize=(8, 8))
#     sharpened = clahe.apply(sharpened)
#
#
#     # sharpened = cv2.cvtColor(sharpened, cv2.COLOR_GRAY2BGR)
#     # sharpened = cv2.GaussianBlur(sharpened, (0, 0), sigma)
#
#     return sharpened



# 定义计算相似度的函数
def calculate_similarity(des1, des2):
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.8 * n.distance:
            good_matches.append(m)

    # 改进相似度计算方法
    similarity = len(good_matches) / min(len(kp1), len(kp2)) * 100  # 使用关键点数量的较小值
    return similarity, good_matches

# 图像对
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

# 遍历图像对
for idx, (img1_path, img2_path) in enumerate(image_pairs):
    # 读取图像
    img1 = cv2.imread(img1_path, cv2.IMREAD_GRAYSCALE)
    img2 = cv2.imread(img2_path, cv2.IMREAD_GRAYSCALE)

    # 检查图像是否读取成功
    if img1 is None or img2 is None:
        print(f"无法读取图像 {img1_path} 或 {img2_path}")
        continue

    # # 使用高斯高通滤波器进行锐化
    # img1_sharpened = gaussian_highpass_sharpen(img1, sigma=1.0)
    # img2_sharpened = gaussian_highpass_sharpen(img2, sigma=1.0)


    # 初始化SIFT检测器并调整参数
    sift = cv2.SIFT_create(
        nfeatures=2000,  # 增加关键点数量
        nOctaveLayers=2,
        contrastThreshold=0.01,  # 降低对比度阈值以增加关键点
        edgeThreshold=25,  # 降低边缘阈值以增加关键点
        sigma=1.1, # 降低高斯核标准差以增加关键点精度

    )

    # 检测关键点和计算描述子
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 计算相似度
    similarity, good_matches = calculate_similarity(des1, des2)

    # 判断是否改装
    if img1_path[:3] == img2_path[:3] and img1_path[3] == img2_path[3]:
        modification_status = "未改装"
    else:
        modification_status = "改装"

    # 输出结果
    print(f"{img1_path} 和 {img2_path} {modification_status}，当前相似度：{similarity:.2f}%",f"{img1_path} 关键点个数：{len(kp1)}",f"{img2_path} 关键点个数：{len(kp2)}")


    # print("-" * 40)

    # 绘制第一对图像的关键点匹配结果
    if idx == 0:  # 只绘制第一对图像的关键点匹配结果
        img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches[:30], None,
                                      matchColor=(0, 255, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
        plt.figure(figsize=(12, 6))
        plt.imshow(img_matches, cmap='gray')
        plt.title(f"关键点匹配结果：{img1_path} 和 {img2_path}")
        plt.show()