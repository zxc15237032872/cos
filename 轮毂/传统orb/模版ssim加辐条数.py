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

import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import argrelextrema
from skimage.metrics import structural_similarity as ssim
from scipy.interpolate import make_interp_spline


def calculate_spoke_number(mask):
    """
    计算图像的辐条数

    :param image_path: 图像的路径
    :return: 辐条数（离 w 最近的整数）或 None（如果未找到大于 0.6 的极大值点）
    """
    # 解决中文显示问题



    # 将图像转换为灰度图，因为 ssim 通常在单通道图像上计算
    gray_image = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    height = 100
    width = 100

    # 存储角度和对应的 SSIM 分数
    angles = []
    ssim_scores = []

    # 遍历可能的旋转角度（比如 0 - 360 度，步长可根据需要调整）
    best_match_score = -float('inf')
    best_rotated_img = None
    angle = 0
    best_angle = 0
    for angle in range(0, 360, 1):
        M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        rotated_img = cv2.warpAffine(gray_image, M, (width, height))
        # 使用 SSIM 衡量旋转后图像与原图像的相似性
        score = ssim(gray_image, rotated_img)
        angles.append(angle)
        ssim_scores.append(score)
        if score > best_match_score:
            best_angle = angle
            best_match_score = score
            best_rotated_img = rotated_img

    # 将角度和 SSIM 分数转换为 numpy 数组
    angles = np.array(angles)
    ssim_scores = np.array(ssim_scores)

    # 绘制平滑曲线
    plt.plot(angles, ssim_scores, label='SSIM 分数')
    # 使用二次样条插值平滑曲线
    xnew = np.linspace(angles.min(), angles.max(), 300)
    spl = make_interp_spline(angles, ssim_scores, k=2)
    ssim_scores_smooth = spl(xnew)
    plt.plot(xnew, ssim_scores_smooth, label='平滑后的 SSIM 分数', linestyle='--')

    # 寻找极大值点和极小值点
    maxima_indices = argrelextrema(ssim_scores_smooth, np.greater)
    minima_indices = argrelextrema(ssim_scores_smooth, np.less)

    maxima_values = ssim_scores_smooth[maxima_indices]
    minima_values = ssim_scores_smooth[minima_indices]

    # 找到从左往右数第一个大于 0.6 的极大值点的索引
    first_valid_maxima_index = None
    for i, value in enumerate(maxima_values):
        if value > 0.6:
            first_valid_maxima_index = i
            break

    if first_valid_maxima_index is not None:
        # 获取对应的角度值
        corresponding_angle = xnew[maxima_indices[0][first_valid_maxima_index]]
        # 计算 w
        w = 360 / corresponding_angle
        # 找到离 w 最近的整数
        nearest_integer = round(w)
        plt.close()  # 关闭绘图窗口
        return nearest_integer
    else:
        plt.close()  # 关闭绘图窗口
        return None

def illumination_compensation1(image):
    """
    对输入图像进行光照补偿，使用CLAHE（对比度受限的自适应直方图均衡化）。

    :param image: 输入的灰度图像
    :return: 光照补偿后的图像
    """
    # 创建 CLAHE 对象
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(9, 9))
    # 应用 CLAHE 进行自适应直方图均衡化
    corrected_image = clahe.apply(image)

    return corrected_image

def illumination_compensation(image):
    # 获取图像的高度和宽度
    height, width = image.shape[:2]
    # 计算图像的中心位置
    center_x, center_y = width // 2, height // 2

    # 初始化一个数组来存储每个角度的平均灰度值
    angle_bins = 360
    angle_avg = np.zeros(angle_bins)

    # 计算每个角度的平均灰度值
    for r in range(1, min(center_x, center_y)):
        for angle in range(angle_bins):
            x = int(center_x + r * np.cos(np.radians(angle)))
            y = int(center_y + r * np.sin(np.radians(angle)))
            if 0 <= x < width and 0 <= y < height:
                angle_avg[angle] += image[y, x]

    # 计算每个角度的平均灰度值
    angle_avg /= min(center_x, center_y) - 1

    # 生成光照补偿图
    compensation_map = np.zeros_like(image)
    for r in range(height):
        for c in range(width):
            dx = c - center_x
            dy = r - center_y
            angle = np.arctan2(dy, dx) * 180 / np.pi
            if angle < 0:
                angle += 360
            angle_index = int(angle)
            compensation_map[r, c] = angle_avg[angle_index]

    # 对光照补偿图进行高斯模糊，平滑光照差异
    compensation_map = cv2.GaussianBlur(compensation_map, (5, 5), 0)

    # 计算平均光照强度
    mean_illumination = np.mean(compensation_map)

    # 进行光照校正
    corrected_image = image * (mean_illumination / compensation_map)
    corrected_image = np.uint8(np.clip(corrected_image, 0, 255))

    # 使用 CLAHE 进一步增强对比度
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    corrected_image = clahe.apply(corrected_image)

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
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.95)

    # 应用 K - Means 算法
    _, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 100, cv2.KMEANS_RANDOM_CENTERS)

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


def process_image_rotation_and_union(image):
    """
    对输入的掩码图像进行旋转操作，并与原图进行并集操作。

    :param image: 输入的掩码图像
    :return: 处理后的并集图像、最佳旋转后的图像和最佳旋转角度
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

    # 实现并集操作
    union_img = cv2.bitwise_or(image, best_rotated_img)

    return union_img, best_rotated_img, best_angle


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

        # 对掩码进行旋转和并集操作
        union_mask1, rotated_mask1_pre, angle1_pre = process_image_rotation_and_union(mask1)
        union_mask1, rotated_mask1_pre, angle1_pre = process_image_rotation_and_union(union_mask1)
        union_mask2, rotated_mask2_pre, angle2_pre = process_image_rotation_and_union(mask2)
        union_mask2, rotated_mask2_pre, angle2_pre = process_image_rotation_and_union(union_mask2)


        # 对每个掩码进行聚类后的处理
        opening_mask1, rotated_mask1, angle1 = process_image_after_clustering(union_mask1)
        opening_mask1, rotated_mask2, angle2 = process_image_after_clustering(opening_mask1)
        opening_mask1, rotated_mask2, angle2 = process_image_after_clustering(opening_mask1)
        opening_mask1, rotated_mask2, angle2 = process_image_after_clustering(opening_mask1)
        opening_mask2, rotated_mask2, angle2 = process_image_after_clustering(union_mask2)
        opening_mask2, rotated_mask2, angle2 = process_image_after_clustering(opening_mask2)
        opening_mask2, rotated_mask2, angle2 = process_image_after_clustering(opening_mask2)
        opening_mask2, rotated_mask2, angle2 = process_image_after_clustering(opening_mask2)
        result1=calculate_spoke_number(opening_mask1)
        result2=calculate_spoke_number(opening_mask2)
        if result1 is not None and result2 is not None:
            if result1 == result2:
                print(f"图片 {image_path1} 和 {image_path2} 的轮毂数相同为: {result1}")
            else:
                print(f"图片 {image_path1}为 {result1}  和 {image_path2}为 {result2} ，轮毂数不同")





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
        weighted_similarity = max_similarity
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