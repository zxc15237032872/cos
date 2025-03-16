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


def run_length_encoding(image):
    """
    对二值图像进行游程编码
    :param image: 输入的二值图像
    :return: 游程编码结果
    """
    rle = []
    for row in image:
        run = 0
        prev_pixel = 0
        for pixel in row:
            if pixel != prev_pixel:
                if run > 0:
                    rle.append(run)
                    rle.append(prev_pixel)
                run = 1
                prev_pixel = pixel
            else:
                run += 1
        if run > 0:
            rle.append(run)
            rle.append(prev_pixel)
    return rle


def rle_similarity(rle1, rle2):
    """
    计算两个游程编码的相似度
    :param rle1: 游程编码 1
    :param rle2: 游程编码 2
    :return: 相似度得分
    """
    min_len = min(len(rle1), len(rle2))
    score = 0
    for i in range(min_len):
        if rle1[i] == rle2[i]:
            score += 1
    return score / min_len if min_len > 0 else 0


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

    rle1 = run_length_encoding(opening_mask1)
    rle2 = run_length_encoding(best_rotated_mask2)
    rle_sim = rle_similarity(rle1, rle2)
    print(f"游程编码相似度为: {rle_sim}")

    final_score = 0.6 * max_similarity + 0.4 * rle_sim

    return final_score, best_angle, opening_mask1, opening_mask2, best_rotated_mask2


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
    similarity, best_angle, opening_mask1, opening_mask2, rotated_mask2 = calculate_similarity(
        image_path1, image_path2)

    print(f"图片 {image_path1} 和 {image_path2} 的综合相似度为: {similarity}")
    print(f"最佳旋转角度为: {best_angle} 度")

    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(opening_mask1, cmap='gray')
    plt.title('第一个图开运算后的 Mask')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(opening_mask2, cmap='gray')
    plt.title('第二个图开运算后的 Mask')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(rotated_mask2, cmap='gray')
    plt.title('第二个图旋转后的 Mask')
    plt.axis('off')

    # plt.show()