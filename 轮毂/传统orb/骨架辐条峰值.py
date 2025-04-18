import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from skimage.metrics import structural_similarity as ssim

# 解决中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
# 显示图像
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.use('TkAgg')


def illumination_compensation(image):
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    corrected_image = clahe.apply(image)
    return corrected_image


def generate_mask(corrected_image, num_clusters=2):
    pixels = corrected_image.reshape((-1, 1))
    pixels = np.float32(pixels)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.95)

    _, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)

    segmented_image = centers[labels.flatten()]

    segmented_image = segmented_image.reshape(corrected_image.shape)

    if centers[0] > centers[1]:
        mask = (segmented_image == centers[0]).astype(np.uint8) * 255
    else:
        mask = (segmented_image == centers[1]).astype(np.uint8) * 255

    return mask


def process_image_after_clustering(image):
    height, width = image.shape

    best_match_score = -float('inf')
    best_rotated_img = None
    angle = 0
    best_angle = 0
    for angle in range(10, 300, 1):
        M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        rotated_img = cv2.warpAffine(image, M, (width, height))
        result = cv2.matchTemplate(image, rotated_img, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        if max_val > best_match_score:
            best_angle = angle
            best_match_score = max_val
            best_rotated_img = rotated_img

    print('Best match score:', best_match_score)
    print('Best angle:', best_angle)

    intersection_img = np.zeros_like(image)
    for y in range(height):
        for x in range(width):
            if image[y, x] != 0 and best_rotated_img[y, x] != 0:
                intersection_img[y, x] = min(image[y, x], best_rotated_img[y, x])

    kernel = np.ones((1, 1), np.uint8)
    opening_img = cv2.morphologyEx(intersection_img.astype(np.uint8), cv2.MORPH_OPEN, kernel)

    return opening_img, best_rotated_img, best_angle


def skeletonize(image):
    """
    对二值图像进行骨架化
    :param image: 输入的二值图像
    :return: 骨架化后的图像
    """
    skeleton = np.zeros(image.shape, np.uint8)
    element = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    while cv2.countNonZero(image) > 0:
        eroded = cv2.erode(image, element)
        temp = cv2.dilate(eroded, element)
        temp = cv2.subtract(image, temp)
        skeleton = cv2.bitwise_or(skeleton, temp)
        image = eroded.copy()
    return skeleton


def skeleton_similarity(skeleton1, skeleton2):
    """
    计算两个骨架图像的相似度
    :param skeleton1: 骨架图像1
    :param skeleton2: 骨架图像2
    :return: 相似度得分
    """
    # 计算骨架图像的轮廓
    contours1, _ = cv2.findContours(skeleton1.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(skeleton2.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_similarity = 0
    for contour1 in contours1:
        max_similarity = 0
        for contour2 in contours2:
            similarity = cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0)
            max_similarity = max(max_similarity, 1 - similarity)  # 转换为相似度，值越大越相似
        total_similarity += max_similarity

    for contour2 in contours2:
        max_similarity = 0
        for contour1 in contours1:
            similarity = cv2.matchShapes(contour2, contour1, cv2.CONTOURS_MATCH_I1, 0)
            max_similarity = max(max_similarity, 1 - similarity)
        total_similarity += max_similarity

    num_contours = len(contours1) + len(contours2)
    average_similarity = total_similarity / num_contours if num_contours > 0 else 0

    return average_similarity


def find_local_maxima(image, neighborhood_size=5):
    """
    找到图像中的局部最大值点
    :param image: 输入图像
    :param neighborhood_size: 邻域大小，用于判断局部最大值
    :return: 局部最大值点的坐标列表
    """
    local_maxima = []
    h, w = image.shape
    for y in range(neighborhood_size // 2, h - neighborhood_size // 2):
        for x in range(neighborhood_size // 2, w - neighborhood_size // 2):
            neighborhood = image[y - neighborhood_size // 2:y + neighborhood_size // 2 + 1,
                                 x - neighborhood_size // 2:x + neighborhood_size // 2 + 1]
            if image[y, x] == np.max(neighborhood):
                local_maxima.append((y, x))
    return local_maxima


def count_spokes_by_rotated_template(skeleton, template):
    """
    通过旋转模板匹配计算辐条个数
    :param skeleton: 骨架化后的图像
    :param template: 未旋转的辐条模板
    :return: 辐条个数
    """
    height, width = template.shape
    center = (width // 2, height // 2)
    spoke_count = 0
    for angle in range(0, 360, 1):  # 逐步旋转模板
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        rotated_template = cv2.warpAffine(template, rotation_matrix, (width, height))
        result = cv2.matchTemplate(skeleton, rotated_template, cv2.TM_CCOEFF_NORMED)

        local_maxima = find_local_maxima(result)
        spoke_count += len(local_maxima)
    # 由于模板旋转过程中可能会重复计数，这里简单去重（实际可能需要更复杂的去重逻辑）
    spoke_count = len(set(local_maxima))
    return spoke_count


def calculate_similarity(image_path1, image_path2):
    image1 = cv2.imread(image_path1, 0)
    image2 = cv2.imread(image_path2, 0)
    size = 80
    image1 = cv2.resize(image1, (size, size), interpolation=cv2.INTER_CUBIC)
    image2 = cv2.resize(image2, (size, size), interpolation=cv2.INTER_CUBIC)

    corrected_image1 = illumination_compensation(image1)
    corrected_image2 = illumination_compensation(image2)

    mask1 = generate_mask(corrected_image1)
    mask2 = generate_mask(corrected_image2)

    opening_mask1, rotated_mask1, angle1 = process_image_after_clustering(mask1)
    opening_mask2, rotated_mask2, angle2 = process_image_after_clustering(mask2)

    height, width = mask2.shape
    center = (width // 2, height // 2)
    max_similarity = -1
    best_angle = 0
    best_rotated_mask2 = opening_mask2.copy()
    for angle in range(0, 360, 1):
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        rotated_mask2 = cv2.warpAffine(opening_mask2, rotation_matrix, (width, height))

        similarity = ssim(opening_mask1, rotated_mask2)
        if similarity > max_similarity:
            max_similarity = similarity
            best_angle = angle
            best_rotated_mask2 = rotated_mask2

    skeleton1 = skeletonize(opening_mask1)
    skeleton2 = skeletonize(best_rotated_mask2)
    skel_sim = skeleton_similarity(skeleton1, skeleton2)

    # 假设这里手动选择了一个未旋转的辐条模板，你可以根据实际情况修改
    spoke_count1 = count_spokes_by_rotated_template(skeleton1, skeleton1)
    spoke_count2 = count_spokes_by_rotated_template(skeleton2, skeleton2)

    final_score = 0.6 * max_similarity + 0.4 * skel_sim

    return final_score, best_angle, opening_mask1, opening_mask2, best_rotated_mask2, skel_sim, skeleton1, skeleton2, spoke_count1, spoke_count2


# 图片对列表
image_pairs = [
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
    similarity, best_angle, opening_mask1, opening_mask2, rotated_mask2, skel_sim, skeleton1, skeleton2, spoke_count1, spoke_count2 = calculate_similarity(
        image_path1, image_path2)

    print(f"图片 {image_path1} 和 {image_path2} 的综合相似度为: {similarity}")
    print(f"骨架相似度为: {skel_sim}")
    print(f"最佳旋转角度为: {best_angle} 度")
    print(f"图片 {image_path1} 的辐条个数为: {spoke_count1}")
    print(f"图片 {image_path2} 的辐条个数为: {spoke_count2}")

    plt.figure(figsize=(18, 5))
    plt.subplot(141)
    plt.imshow(opening_mask1, cmap='gray')
    plt.title('第一个图开运算后的 Mask')
    plt.axis('off')

    plt.subplot(142)
    plt.imshow(opening_mask2, cmap='gray')
    plt.title('第二个图开运算后的 Mask')
    plt.axis('off')

    plt.subplot(143)
    plt.imshow(rotated_mask2, cmap='gray')
    plt.title('第二个图旋转后的 Mask')
    plt.axis('off')

    plt.subplot(144)
    plt.imshow(skeleton1, cmap='gray')
    plt.title('第一个图骨架化结果')
    plt.axis('off')
    #
    # plt.show()

    plt.figure(figsize=(18, 5))
    plt.subplot(141)
    plt.imshow(opening_mask1, cmap='gray')
    plt.title('第一个图开运算后的 Mask')
    plt.axis('off')

    plt.subplot(142)
    plt.imshow(opening_mask2, cmap='gray')
    plt.title('第二个图开运算后的 Mask')
    plt.axis('off')

    plt.subplot(143)
    plt.imshow(rotated_mask2, cmap='gray')
    plt.title('第二个图旋转后的 Mask')
    plt.axis('off')

    plt.subplot(144)
    plt.imshow(skeleton2, cmap='gray')
    plt.title('第二个图骨架化结果')
    plt.axis('off')

    # plt.show()