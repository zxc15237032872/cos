import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from skimage.metrics import structural_similarity as ssim

# 解决中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 显示图像
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
matplotlib.use('TkAgg')


def illumination_compensation(image):
    # 创建 CLAHE 对象
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # 应用 CLAHE 进行自适应直方图均衡化
    corrected_image = clahe.apply(image)
    return corrected_image


def generate_mask(corrected_image, num_clusters=2):
    # 将图像转换为一维数组
    pixels = corrected_image.reshape((-1, 1))
    pixels = np.float32(pixels)

    # 定义 K - Means 算法的终止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # 应用 K - Means 算法
    _, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 将中心值转换为整数
    centers = np.uint8(centers)

    # 根据标签获取每个像素的聚类中心值
    segmented_image = centers[labels.flatten()]

    # 重新调整形状以匹配原始图像
    segmented_image = segmented_image.reshape(corrected_image.shape)

    # 假设轮毂对应的聚类中心值较大，获取 mask
    if centers[0] > centers[1]:
        mask = (segmented_image == centers[0]).astype(np.uint8) * 255
    else:
        mask = (segmented_image == centers[1]).astype(np.uint8) * 255

    return mask


def process_image_after_clustering(image):
    height, width = image.shape

    # 遍历可能的旋转角度（比如0 - 360度，步长可根据需要调整）
    best_match_score = -float('inf')
    best_rotated_img = None
    angle = 0
    best_angle = 0
    for angle in range(10, 300, 1):
        M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        rotated_img = cv2.warpAffine(image, M, (width, height))
        # 使用模板匹配衡量旋转后图像与原图像的相似性
        result = cv2.matchTemplate(image, rotated_img, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if max_val > best_match_score:
            best_angle = angle
            best_match_score = max_val
            best_rotated_img = rotated_img

    print('Best match score:', best_match_score)
    print('Best angle:', best_angle)

    # 实现类似交集操作
    intersection_img = np.zeros_like(image)
    for y in range(height):
        for x in range(width):
            if image[y, x] != 0 and best_rotated_img[y, x] != 0:
                intersection_img[y, x] = min(image[y, x], best_rotated_img[y, x])

    return intersection_img, best_rotated_img, best_angle


def calculate_similarity(image_path1, image_path2):
    # 读取图像
    image1 = cv2.imread(image_path1, 0)
    image2 = cv2.imread(image_path2, 0)
    size = 100
    # 调整图像大小
    image1 = cv2.resize(image1, (size, size), interpolation=cv2.INTER_CUBIC)
    image2 = cv2.resize(image2, (size, size), interpolation=cv2.INTER_CUBIC)

    # 光照补偿
    corrected_image1 = illumination_compensation(image1)
    corrected_image2 = illumination_compensation(image2)

    # 生成掩码
    mask1 = generate_mask(corrected_image1)
    mask2 = generate_mask(corrected_image2)

    # 对每个掩码进行聚类后的处理
    intersection_mask1, rotated_mask1, angle1 = process_image_after_clustering(mask1)
    intersection_mask2, rotated_mask2, angle2 = process_image_after_clustering(mask2)

    # 考虑旋转对称，尝试不同旋转角度
    height, width = mask2.shape
    center = (width // 2, height // 2)
    max_similarity = -1
    best_angle = 0
    best_rotated_mask2 = intersection_mask2.copy()  # 使用交集后的掩码
    for angle in range(0, 360, 1):  # 以 1 度为步长旋转
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        rotated_mask2 = cv2.warpAffine(intersection_mask2, rotation_matrix, (width, height))

        # 使用 SSIM 计算相似度，使用交集后的掩码
        similarity = ssim(intersection_mask1, rotated_mask2)
        if similarity > max_similarity:
            max_similarity = similarity
            best_angle = angle
            best_rotated_mask2 = rotated_mask2

    return max_similarity, best_angle, intersection_mask1, intersection_mask2, best_rotated_mask2


# 图片对列表
image_pairs = [
    # ('007A.png', '007A1.png'),
    # ('010A.png', '010A1.png'),
    ('011A.png', '011A1.png'),
    ('012A.png', '012A1.png'),
    ('007A.png', '007A.png'),
    ('006A.png', '006A1.png'),
    ('009A.png', '009B.png'),
    ('005A.png', '004A1.png'),
    ('006A.png', '005A.png'),
    ('005A.png', '005B.png'),
    ('006A.png', '004A1.png'),
    ('004A.png', '005A.png'),
    ('005A.png', '006A.png')
]

# 遍历图片对
for pair in image_pairs:
    image_path1, image_path2 = pair
    # 计算相似度、最佳旋转角度，获取掩码和旋转后的掩码
    similarity, best_angle, intersection_mask1, intersection_mask2, rotated_mask2 = calculate_similarity(image_path1, image_path2)

    print(f"图片 {image_path1} 和 {image_path2} 的相似度为: {similarity}")
    print(f"最佳旋转角度为: {best_angle} 度")

    # 显示第一个图的交集后的 mask、第二个图的交集后的 mask 和第二个图旋转之后的 mask
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(intersection_mask1, cmap='gray')
    plt.title('第一个图交集后的 Mask')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(intersection_mask2, cmap='gray')
    plt.title('第二个图交集后的 Mask')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(rotated_mask2, cmap='gray')
    plt.title('第二个图旋转后的 Mask')
    plt.axis('off')

    plt.show()