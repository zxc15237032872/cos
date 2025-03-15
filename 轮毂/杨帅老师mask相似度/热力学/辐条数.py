import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置为支持中文的字体，如黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题
import cv2
import numpy as np
from scipy.ndimage import rotate as ndimage_rotate


def preprocess_image(image):
    """
    对输入的图像进行预处理，包括灰度化、滤波、阈值处理、形态学操作、旋转校正和裁剪
    """
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 高斯滤波减少噪声
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 自适应阈值处理
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 51, 5)
    # 形态学闭运算，填充小空洞
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # 找到轮毂的旋转角度并校正
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        rect = cv2.minAreaRect(largest_contour)
        angle = rect[2]
        rotated_image = ndimage_rotate(image, angle, reshape=False)
        rotated_mask = ndimage_rotate(mask, angle, reshape=False)
    else:
        rotated_image = image
        rotated_mask = mask

    # 找到轮毂轮廓并裁剪
    contours, _ = cv2.findContours(rotated_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)
        cropped_image = rotated_image[y:y + h, x:x + w]
        cropped_mask = rotated_mask[y:y + h, x:x + w]
    else:
        cropped_image = rotated_image
        cropped_mask = rotated_mask

    return cropped_image, cropped_mask


def count_spokes(mask):
    """
    计算轮毂掩码中的辐条数量
    """
    h, w = mask.shape
    center = (w // 2, h // 2)
    max_radius = min(center[0], center[1]) - 1

    # 极坐标变换
    polar = cv2.warpPolar(
        src=mask,
        dsize=(360, max_radius),
        center=center,
        maxRadius=max_radius,
        flags=cv2.WARP_POLAR_LINEAR
    )

    # 对极坐标图像进行垂直投影
    projection = np.sum(polar, axis=0)
    # 寻找投影中的峰值，认为是辐条的位置
    peaks = []
    threshold = np.mean(projection) * 1.2  # 降低峰值阈值
    for i in range(1, len(projection) - 1):
        if projection[i] > projection[i - 1] and projection[i] > projection[i + 1] and projection[i] > threshold:
            peaks.append(i)
    return len(peaks)


def extract_sectors(mask, spoke_count):
    """
    以辐条为中心划分扇区
    """
    h, w = mask.shape
    center = (w // 2, h // 2)
    max_radius = min(center[0], center[1]) - 1

    # 极坐标变换
    polar = cv2.warpPolar(
        src=mask,
        dsize=(360, max_radius),
        center=center,
        maxRadius=max_radius,
        flags=cv2.WARP_POLAR_LINEAR
    )

    # 计算扇区角度
    sector_angle = 360 // spoke_count
    sectors = []
    for i in range(spoke_count):
        start_angle = i * sector_angle
        end_angle = (i + 1) * sector_angle
        sector = polar[start_angle:end_angle, :]
        sectors.append(sector)
    return sectors


def find_best_sector(sectors):
    """
    找到最接近其他扇区的扇区
    """
    num_sectors = len(sectors)
    best_sector_index = 0
    best_similarity = float('inf')

    for i in range(num_sectors):
        total_similarity = 0
        for j in range(num_sectors):
            if i != j:
                similarity = cv2.matchShapes(
                    cv2.findContours(sectors[i].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
                    cv2.findContours(sectors[j].astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0],
                    cv2.CONTOURS_MATCH_I3, 0
                )
                total_similarity += similarity
        average_similarity = total_similarity / (num_sectors - 1)
        if average_similarity < best_similarity:
            best_similarity = average_similarity
            best_sector_index = i
    return sectors[best_sector_index]


def morphological_matching(mask1, mask2):
    """
    进行形态学匹配
    """
    # 计算两个掩码的形态学差异
    xor_result = cv2.bitwise_xor(mask1, mask2)
    # 计算差异区域的像素数量
    diff_pixels = cv2.countNonZero(xor_result)
    # 计算总像素数量
    total_pixels = mask1.size
    # 计算相似度得分
    similarity_score = 1 - (diff_pixels / total_pixels)
    return similarity_score


image_pairs = [
    ('007A.png', '007A1.png'),
    ('007A.png', '007A.png'),
    ('006A.png', '006A1.png'),
    ('004A.png', '004A1.png'),
    ('005A.png', '004A1.png'),
    ('006A.png', '005A.png'),
    ('005A.png', '005B.png'),
    ('006A.png', '004A1.png'),
    ('004A.png', '005A.png'),
    ('005A.png', '006A.png')
]

similarities = []
is_modified_list = []

for pair in image_pairs:
    img_path1, img_path2 = pair
    # 读取图像
    image1 = cv2.imread(img_path1)
    image2 = cv2.imread(img_path2)

    if image1 is None or image2 is None:
        print(f"无法读取图像: {img_path1} 或 {img_path2}")
        continue

    # 预处理图像
    _, mask1 = preprocess_image(image1)
    _, mask2 = preprocess_image(image2)

    # 计算辐条数量
    spokes1 = count_spokes(mask1)
    spokes2 = count_spokes(mask2)

    # 判断是否改装（辐条数量不同则认为改装）
    is_modified = spokes1 != spokes2

    if spokes1 > 0 and spokes2 > 0:
        # 提取扇区
        sectors1 = extract_sectors(mask1, spokes1)
        sectors2 = extract_sectors(mask2, spokes2)

        # 找到最佳扇区
        best_sector1 = find_best_sector(sectors1)
        best_sector2 = find_best_sector(sectors2)

        # 进行形态学匹配
        similarity = morphological_matching(best_sector1.astype(np.uint8), best_sector2.astype(np.uint8))
    else:
        similarity = None

    similarities.append(similarity)
    is_modified_list.append(is_modified)

    print(f"图像对 ({img_path1}, {img_path2})")
    print(f"第一张图像的辐条数量: {spokes1}")
    print(f"第二张图像的辐条数量: {spokes2}")
    print(f"是否改装: {'是' if is_modified else '否'}")
    if similarity is not None:
        print(f"最佳扇区相似度: {similarity}")
    else:
        print("无法计算相似度，辐条数量不足。")
    print("-" * 50)

# 绘制第一组图像的相似度结果（这里仅为示例，你可以根据需求修改）
if similarities and similarities[0] is not None:
    plt.figure()
    plt.bar(['相似度'], [similarities[0]])
    plt.title(f"图像对 ({image_pairs[0][0]}, {image_pairs[0][1]}) 相似度")
    plt.ylabel('相似度（值越大越相似）')
    plt.show()