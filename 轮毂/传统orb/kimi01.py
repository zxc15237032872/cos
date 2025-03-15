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

    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good_matches.append(m)

    # 使用关键点数量的较小值计算相似度
    similarity = len(good_matches) / min(len(des1), len(des2)) * 100
    return similarity, good_matches

# 图像对
image_pairs = [
    ('007A.png', '007A1.png'),  # 未改装
    ('007A.png', '007A.png'),   # 未改装
    ('006A.png', '006A1.png'),  # 未改装
    ('005A.png', '004A1.png'),  # 改装
    ('006A.png', '005A.png'),   # 改装
    ('005A.png', '005B.png'),   # 改装
    ('006A.png', '004A1.png'),  # 改装
    ('004A.png', '005A.png'),   # 改装
    ('005A.png', '006A.png')    # 改装
]

# 初始化最佳参数和相似度差距
best_threshold = 0.01
max_similarity_gap = 0

# 遍历不同的 contrastThreshold 值
for threshold in np.arange(0.01, 0.21, 0.01):
    similarities = {"unmodified": [], "modified": []}  # 用于存储未改装和改装的相似度

    # 遍历图像对
    for img1_path, img2_path in image_pairs:
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

        # 初始化 SIFT 检测器并调整参数
        sift = cv2.SIFT_create(
            nfeatures=2000,
            nOctaveLayers=2,
            contrastThreshold=threshold,
            edgeThreshold=16,
            sigma=1.1
        )

        # 检测关键点和计算描述子
        kp1, des1 = sift.detectAndCompute(img1_sharpened, None)
        kp2, des2 = sift.detectAndCompute(img2_sharpened, None)

        # 计算相似度
        similarity, _ = calculate_similarity(des1, des2)

        # 判断是否改装
        if img1_path[:3] == img2_path[:3] and img1_path[3] == img2_path[3]:
            modification_status = "unmodified"
        else:
            modification_status = "modified"

        # 存储相似度
        similarities[modification_status].append(similarity)

    # 计算未改装和改装的平均相似度
    avg_unmodified_similarity = np.mean(similarities["unmodified"])
    avg_modified_similarity = np.mean(similarities["modified"])

    # 计算相似度差距
    similarity_gap = abs(avg_unmodified_similarity - avg_modified_similarity)

    # 更新最佳的 contrastThreshold 值
    if similarity_gap > max_similarity_gap:
        max_similarity_gap = similarity_gap
        best_threshold = threshold

    print(f"contrastThreshold = {threshold:.2f}")
    print(f"未改装平均相似度：{avg_unmodified_similarity:.2f}%")
    print(f"改装平均相似度：{avg_modified_similarity:.2f}%")
    print(f"相似度差距：{similarity_gap:.2f}")
    print("-" * 40)

print(f"最佳的 contrastThreshold 值为：{best_threshold:.2f}")
print(f"此时的最大相似度差距为：{max_similarity_gap:.2f}")