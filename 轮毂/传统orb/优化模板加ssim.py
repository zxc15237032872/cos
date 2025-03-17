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
    """
    对输入图像进行光照补偿，使用CLAHE（对比度受限的自适应直方图均衡化）。

    :param image: 输入的灰度图像
    :return: 光照补偿后的图像
    """
    # 创建 CLAHE 对象


    clahe = cv2.createCLAHE(clipLimit=1.1, tileGridSize=(16, 16))
    # 应用 CLAHE 进行自适应直方图均衡化
    corrected_image = clahe.apply(image)

    return corrected_image


def generate_mask(corrected_image, num_clusters=2):
    """
    使用K-Means算法生成图像的掩码。

    :param corrected_image: 光照补偿后的图像
    :param num_clusters: 聚类的数量，默认为2
    :return: 生成的掩码图像
    """
    # 将图像转换为一维数组
    pixels = corrected_image.reshape((-1, 1))
    pixels = np.float32(pixels)

    # 定义 K - Means 算法的终止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.2)

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
    # cv2.imshow('K-Means 聚类', segmented_image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()

    return mask


def process_image_after_clustering(image):
    """
    对聚类后的掩码图像进行处理，包括旋转和交集操作。

    :param image: 输入的掩码图像
    :return: 处理后的交集图像、最佳旋转后的图像和最佳旋转角度
    """
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
    """
    计算两张图像的加权相似度，包括光照补偿、掩码生成、旋转处理和相似度计算。

    :param image_path1: 第一张图像的路径
    :param image_path2: 第二张图像的路径
    :return: 加权后的相似度、最佳旋转角度、第一张图像的掩码、第二张图像的掩码、第二张图像旋转后的掩码和模板匹配分数
    """
    try:
        # 读取图像
        image1 = cv2.imread(image_path1, 0)
        image2 = cv2.imread(image_path2, 0)
        if image1 is None or image2 is None:
            raise FileNotFoundError(f"无法读取图像文件: {image_path1} 或 {image_path2}")

        size = 200
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
        opening_mask1, rotated_mask1, angle1 = process_image_after_clustering(mask1)
        opening_mask2, rotated_mask2, angle2 = process_image_after_clustering(mask2)

        # 考虑旋转对称，尝试不同旋转角度
        height, width = mask2.shape
        center = (width // 2, height // 2)
        max_similarity = -1
        best_angle = 0
        best_rotated_mask2 = opening_mask2.copy()
        for angle in range(0, 360, 1):  # 以 1 度为步长旋转
            rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
            rotated_mask2 = cv2.warpAffine(opening_mask2, rotation_matrix, (width, height))

            # 使用 SSIM 计算相似度
            similarity = ssim(opening_mask1, rotated_mask2)
            if similarity > max_similarity:
                max_similarity = similarity
                best_angle = angle
                best_rotated_mask2 = rotated_mask2

        # 模板匹配
        template_match_result = cv2.matchTemplate(opening_mask1, best_rotated_mask2, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(template_match_result)

        # 计算加权后的相似度
        weighted_similarity = 0.4 * max_similarity + 0.6 * max_val
        return weighted_similarity, best_angle, opening_mask1, opening_mask2, best_rotated_mask2, max_val
    except Exception as e:
        print(f"计算相似度时出错: {e}")
        return None, None, None, None, None, None


# 图片对列表
image_pairs = [
    # ('007A.png', '007A1.png'),
    # ('010A.png', '010A1.png'),
    ('005A.png', '0003A.png'),
    ('011A.png', '011A1.png'),
    ('012A.png', '012A1.png'),
    ('007A.png', '007A.png'),
    ('006A.png', '006A1.png'),
    ('011A.png', '009A.png'),
    ('006A.png', '005B.png'),
    ('012A.png', '005B.png'),
    ('006A.png', '004A1.png'),
    ('009A.png', '005B.png'),
    ('005B.png', '006A.png')
]

# 遍历图片对
for pair in image_pairs:
    image_path1, image_path2 = pair
    # 计算相似度、最佳旋转角度，获取掩码和旋转后的掩码以及模板匹配分数
    weighted_similarity, best_angle, opening_mask1, opening_mask2, rotated_mask2, template_match_score = calculate_similarity(
        image_path1, image_path2)

    if weighted_similarity is not None:
        print(f"图片 {image_path1} 和 {image_path2} 的加权后相似度为: {weighted_similarity}")
        print(f"最佳旋转角度为: {best_angle} 度")
        print(f"图片 {image_path1} 和 {image_path2} 的模板匹配分数为: {template_match_score}")

        # 显示第一个图开运算后的 mask、第二个图开运算后的 mask 和第二个图旋转之后的 mask
        plt.figure(figsize=(9, 3))
        plt.subplot(131)
        plt.imshow(opening_mask1, cmap='gray')
        plt.title(image_path1 + '的 Mask')
        plt.axis('off')

        plt.subplot(132)
        plt.imshow(opening_mask2, cmap='gray')
        plt.title(image_path2 + '的 Mask')  # 修正此处的标题为 image_path2
        plt.axis('off')

        plt.subplot(133)
        plt.imshow(rotated_mask2, cmap='gray')
        plt.title(image_path2 + '旋转后的 Mask')
        plt.axis('off')

        plt.show()