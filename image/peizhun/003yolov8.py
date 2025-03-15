import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

# 设置中文字体和解决负号显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

# 使用 K-Means 聚类分割车辆
def segment_car(image):
    # 将图像转换为一维数组
    pixels = image.reshape((-1, 3)).astype(np.float32)
    # 定义 K-Means 参数
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2  # 聚类数量
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    # 找到占比最大的簇
    unique, counts = np.unique(labels, return_counts=True)
    dominant_cluster = unique[np.argmax(counts)]
    # 创建掩码
    mask = (labels == dominant_cluster).reshape(image.shape[:2]).astype(np.uint8)
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(contours) > 0:
        # 找到最大的轮廓
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        car_image = image[y:y + h, x:x + w]
        return car_image
    return image

# 图像配准函数，使用 SIFT 特征
def register_images(img1, img2):
    # 创建 SIFT 特征检测器
    sift = cv2.SIFT_create(
        nfeatures=2000,  # 增加关键点数量
        nOctaveLayers=2,
        contrastThreshold=0.1,  # 降低对比度阈值以增加关键点
        edgeThreshold=25,  # 降低边缘阈值以增加关键点
        sigma=1.1, # 降低高斯核标准差以增加关键点精度

    )
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    # 创建 FLANN 匹配器
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)

    # 筛选好的匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)

    # 输出关键点个数
    print(f"图像 1 关键点个数: {len(kp1)}")
    print(f"图像 2 关键点个数: {len(kp2)}")
    print(f"匹配的关键点对数: {len(good_matches)}")

    # 绘制匹配线
    img_matches = cv2.drawMatches(img1, kp1, img2, kp2, good_matches, None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # 获取匹配点的坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

    # 计算仿射变换矩阵
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    # 应用仿射变换
    h, w = img1.shape[:2]
    registered_img = cv2.warpAffine(img2, M, (w, h))

    return registered_img, img_matches

# 主函数
def main():
    # 输入图片路径
    image_path1 = 'che001.png'
    image_path2 = 'che002.png'
    # 读取图片
    img1 = cv2.imread(image_path1)
    img2 = cv2.imread(image_path2)

    # 分割车辆
    car1 = segment_car(img1)
    car2 = segment_car(img2)

    # 图像配准
    registered_car2, img_matches = register_images(car1, car2)

    # 绘制四张图
    plt.figure(figsize=(20, 5))
    plt.subplot(141)
    plt.imshow(cv2.cvtColor(img1, cv2.COLOR_BGR2RGB))
    plt.title('车 A')
    plt.axis('off')
    plt.subplot(142)
    plt.imshow(cv2.cvtColor(img2, cv2.COLOR_BGR2RGB))
    plt.title('车 B (原始)')
    plt.axis('off')
    plt.subplot(143)
    plt.imshow(cv2.cvtColor(registered_car2, cv2.COLOR_BGR2RGB))
    plt.title('车 B (配准后)')
    plt.axis('off')
    plt.subplot(144)
    plt.imshow(cv2.cvtColor(img_matches, cv2.COLOR_BGR2RGB))
    plt.title('关键点匹配')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()