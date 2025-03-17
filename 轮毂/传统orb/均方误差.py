import cv2
import numpy as np

def rotate_image(image, angle):
    """
    对图像进行旋转操作
    :param image: 输入的图像
    :param angle: 旋转角度
    :return: 旋转后的图像
    """
    height, width = image.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height))
    _, rotated_image = cv2.threshold(rotated_image, 127, 255, cv2.THRESH_BINARY)
    return rotated_image

def jaccard_similarity(template, image):
    """
    计算两张二值图像的杰卡德相似度
    :param template: 模板图像
    :param image: 待比较图像
    :return: 杰卡德相似度
    """
    intersection = np.logical_and(template, image)
    union = np.logical_or(template, image)
    jaccard = np.sum(intersection) / np.sum(union)
    return jaccard

def calculate_rotational_similarity(image_path1, image_path2, step=1):
    """
    计算两张近似旋转对称的二值图片的最大相似度
    :param image_path1: 第一张图片的路径
    :param image_path2: 第二张图片的路径
    :param step: 旋转角度的步长
    :return: 最大相似度（百分数形式）
    """
    # 读取图片并转换为二值图
    image1 = cv2.imread(image_path1, 0)
    image1 = cv2.resize(image1, (200, 200))
    _, image1 = cv2.threshold(image1, 127, 255, cv2.THRESH_BINARY)
    image2 = cv2.imread(image_path2, 0)
    image2 = cv2.resize(image2, (200, 200))
    _, image2 = cv2.threshold(image2, 127, 255, cv2.THRESH_BINARY)

    max_similarity = 0
    # 遍历不同的旋转角度
    for angle in range(0, 360, step):
        rotated_image = rotate_image(image2, angle)
        similarity = jaccard_similarity(image1, rotated_image)
        if similarity > max_similarity:
            max_similarity = similarity

    # 将相似度转换为百分数形式
    similarity_percentage = max_similarity * 100
    return similarity_percentage

# 示例调用
image_path1 = 'img_3.png'
image_path2 = 'img_4.png'

similarity = calculate_rotational_similarity(image_path1, image_path2)
print(f"两张图片的近似旋转对称相似度为: {similarity:.2f}%")