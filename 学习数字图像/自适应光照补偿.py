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


def calculate_similarity(image_path1, image_path2):
    # 读取图像
    image1 = cv2.imread(image_path1, 0)
    image2 = cv2.imread(image_path2, 0)
    # 调整图像大小
    image1 = cv2.resize(image1, (256, 256))
    image2 = cv2.resize(image2, (256, 256))


    # 光照补偿
    corrected_image1 = illumination_compensation(image1)
    corrected_image2 = illumination_compensation(image2)

    # 生成掩码
    mask1 = generate_mask(corrected_image1)
    mask2 = generate_mask(corrected_image2)

    # 考虑旋转对称，尝试不同旋转角度
    height, width = mask2.shape
    center = (width // 2, height // 2)
    max_similarity = -1
    best_angle = 0
    best_rotated_mask2 = mask2.copy()
    for angle in range(0, 360, 10):  # 以 10 度为步长旋转
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        rotated_mask2 = cv2.warpAffine(mask2, rotation_matrix, (width, height))

        # 使用 SSIM 计算相似度
        similarity = ssim(mask1, rotated_mask2)
        if similarity > max_similarity:
            max_similarity = similarity
            best_angle = angle
            best_rotated_mask2 = rotated_mask2

    return max_similarity, best_angle, mask1, mask2, best_rotated_mask2


# 示例图片路径
image_path1 = '005A.png'
image_path2 = '005A.png'

# 计算相似度、最佳旋转角度，获取掩码和旋转后的掩码
similarity, best_angle, mask1, mask2, rotated_mask2 = calculate_similarity(image_path1, image_path2)

print(f"两张图片的相似度为: {similarity}")
print(f"最佳旋转角度为: {best_angle} 度")

# 显示第一个图的 mask、第二个图的 mask 和第二个图旋转之后的 mask
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(mask1, cmap='gray')
plt.title('第一个图的 Mask')
plt.axis('off')

plt.subplot(132)
plt.imshow(mask2, cmap='gray')
plt.title('第二个图的 Mask')
plt.axis('off')

plt.subplot(133)
plt.imshow(rotated_mask2, cmap='gray')
plt.title('第二个图旋转后的 Mask')
plt.axis('off')

plt.show()