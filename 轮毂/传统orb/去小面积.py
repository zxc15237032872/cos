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

    mask=cv2.GaussianBlur(mask,(3,3),0)
    # 显示原始图像、CLAHE 图像、分割图像和掩码
    # plt.figure(figsize=(5, 5))
    # plt.subplot(111)
    # plt.imshow(mask)
    # plt.title('原始图像')
    # plt.show()

    return mask


def remove_insignificant_structures(mask, min_area_threshold=1, kernel_size=(1, 1)):
    # 腐蚀操作，缩小区域
    kernel = np.ones(kernel_size, np.uint8)
    eroded_mask = cv2.erode(mask, kernel, iterations=1)

    # 查找轮廓
    contours, _ = cv2.findContours(eroded_mask.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 创建一个新的掩码，初始化为全零
    new_mask = np.zeros_like(mask)

    # 遍历每个轮廓
    for contour in contours:
        area = cv2.contourArea(contour)
        # new_mask1 = np.zeros_like(mask)
        # cv2.drawContours(new_mask1, [contour], 0, 255, 1)
        # plt.figure(figsize=(5, 5))
        # plt.subplot(111)
        # plt.imshow(new_mask1, cmap='gray')
        # plt.title('原始dilated图像')
        # plt.show()



        # 如果面积大于阈值，则保留该轮廓对应的区域

        if area > min_area_threshold:
            cv2.drawContours(new_mask, [contour], 0, 255, 1)

    # 膨胀操作，恢复区域大小
    dilated_mask = cv2.dilate(new_mask, kernel, iterations=1)
    # plt.figure(figsize=(15, 5))
    # plt.subplot(111)
    # plt.imshow(dilated_mask, cmap='gray')
    # plt.title('原始dilated图像')
    # plt.show()

    return dilated_mask


def calculate_similarity(image_path1, image_path2):
    # 读取图像
    image1 = cv2.imread(image_path1, 0)
    image2 = cv2.imread(image_path2, 0)
    size = 200

    # 对原图进行高斯滤波
    image1 = cv2.GaussianBlur(image1, (3, 3), 0)
    image2 = cv2.GaussianBlur(image2, (3, 3), 0)

    # 调整图像大小
    image1 = cv2.resize(image1, (size, size), interpolation=cv2.INTER_CUBIC)
    image2 = cv2.resize(image2, (size, size), interpolation=cv2.INTER_CUBIC)

    # 光照补偿
    corrected_image1 = illumination_compensation(image1)
    corrected_image2 = illumination_compensation(image2)

    # 生成掩码
    mask1 = generate_mask(corrected_image1)
    mask2 = generate_mask(corrected_image2)

    # 去除无关紧要的结构
    mask1 = remove_insignificant_structures(mask1)
    mask2 = remove_insignificant_structures(mask2)

    # 查找轮廓
    contours1, _ = cv2.findContours(mask1.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours2, _ = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 创建彩色图像用于绘制轮廓
    contour_image1 = cv2.cvtColor(mask1, cv2.COLOR_GRAY2BGR)
    contour_image2 = cv2.cvtColor(mask2, cv2.COLOR_GRAY2BGR)

    # 绘制轮廓
    cv2.drawContours(contour_image1, contours1, -1, (0, 255, 0), 1)
    cv2.drawContours(contour_image2, contours2, -1, (0, 255, 0), 1)

    # 考虑旋转对称，尝试不同旋转角度
    height, width = mask2.shape
    center = (width // 2, height // 2)
    max_similarity = -1
    best_angle = 0
    best_rotated_mask2 = mask2.copy()
    for angle in range(0, 360, 1):  # 以 1 度为步长旋转
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        rotated_mask2 = cv2.warpAffine(mask2, rotation_matrix, (width, height))

        # 使用 SSIM 计算相似度
        similarity = ssim(mask1, rotated_mask2)
        if similarity > max_similarity:
            max_similarity = similarity
            best_angle = angle
            best_rotated_mask2 = rotated_mask2

    return max_similarity, best_angle, contour_image1, contour_image2, best_rotated_mask2


# 图片对列表
image_pairs = [
    ('006A.png', '005A.png'),
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
    similarity, best_angle, contour_image1, contour_image2, rotated_mask2 = calculate_similarity(image_path1, image_path2)

    print(f"图片 {image_path1} 和 {image_path2} 的相似度为: {similarity}")
    print(f"最佳旋转角度为: {best_angle} 度")

    # 显示第一个图的轮廓、第二个图的轮廓和第二个图旋转之后的 mask
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(contour_image1, cv2.COLOR_BGR2RGB))
    plt.title('第一个图的轮廓')
    plt.axis('off')

    plt.subplot(132)
    plt.imshow(cv2.cvtColor(contour_image2, cv2.COLOR_BGR2RGB))
    plt.title('第二个图的轮廓')
    plt.axis('off')

    plt.subplot(133)
    plt.imshow(rotated_mask2, cmap='gray')
    plt.title('第二个图旋转后的 Mask')
    plt.axis('off')

    plt.show()