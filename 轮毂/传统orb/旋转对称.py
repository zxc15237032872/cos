import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from skimage.metrics import structural_similarity as ssim

# 配置中文字体
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.use('TkAgg')


def illumination_compensation(image):
    """改进的自适应直方图均衡化"""
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(16, 16))
    return clahe.apply(image)


def generate_mask(corrected_image):
    """改进的掩码生成方法（基于Otsu阈值）"""
    _, mask = cv2.threshold(corrected_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask


def refine_mask(mask):
    """改进的掩码精炼流程"""
    # 形态学操作（先闭后开）
    kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_close, iterations=2)

    kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    opened = cv2.morphologyEx(closed, cv2.MORPH_OPEN, kernel_open, iterations=2)

    # 去除小区域并填充孔洞
    opened = remove_small_areas(opened, min_area_threshold=100)
    opened = fill_holes(opened)
    return opened


def remove_small_areas(mask, min_area_threshold=100):
    """改进的小区域去除"""
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    new_mask = np.zeros_like(mask)
    for contour in contours:
        if cv2.contourArea(contour) > min_area_threshold:
            cv2.drawContours(new_mask, [contour], -1, 255, -1)
    return new_mask


def fill_holes(mask):
    """填充掩码中的孔洞"""
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    filled = np.zeros_like(mask)
    cv2.drawContours(filled, contours, -1, 255, -1)
    return filled


def align_center(mask):
    """中心对齐处理"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask

    # 计算最大轮廓的质心
    M = cv2.moments(contours[0])
    if M["m00"] == 0:
        return mask
    cx = int(M["m10"] / M["m00"])
    cy = int(M["m01"] / M["m00"])

    # 平移图像使质心居中
    rows, cols = mask.shape
    dx = cols // 2 - cx
    dy = rows // 2 - cy
    M = np.float32([[1, 0, dx], [0, 1, dy]])
    return cv2.warpAffine(mask, M, (cols, rows))


def calculate_similarity(image_path1, image_path2):
    """改进的相似度计算流程"""
    # 图像预处理
    size = 300  # 增大处理尺寸
    image1 = cv2.resize(cv2.imread(image_path1, 0), (size, size))
    image2 = cv2.resize(cv2.imread(image_path2, 0), (size, size))

    # 光照补偿
    corrected1 = illumination_compensation(image1)
    corrected2 = illumination_compensation(image2)

    # 生成并优化掩码
    mask1 = generate_mask(corrected1)
    mask2 = generate_mask(corrected2)

    mask1 = refine_mask(mask1)
    mask2 = refine_mask(mask2)

    # 中心对齐
    mask1 = align_center(mask1)
    mask2 = align_center(mask2)

    # 旋转匹配优化
    best_similarity = 0
    best_angle = 0
    best_rotated = mask2

    # 优化旋转步长（粗调+精调）
    for coarse_angle in range(0, 360, 10):
        for fine_angle in range(coarse_angle - 5, coarse_angle + 5):
            rotated = rotate_image(mask2, fine_angle)
            similarity = enhanced_similarity(mask1, rotated)
            if similarity > best_similarity:
                best_similarity = similarity
                best_angle = fine_angle
                best_rotated = rotated

    return best_similarity * 100, best_angle, mask1, mask2, best_rotated


def rotate_image(image, angle):
    """改进的旋转方法（保持内容不丢失）"""
    h, w = image.shape
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)


def enhanced_similarity(mask1, mask2):
    """改进的相似度评估（综合SSIM和形状匹配）"""
    # SSIM相似度
    ssim_score = ssim(mask1, mask2)

    # 形状匹配得分
    contours1 = get_main_contours(mask1)
    contours2 = get_main_contours(mask2)

    shape_score = 0
    min_len = min(len(contours1), len(contours2))
    if min_len == 0:
        return 0

    for i in range(min_len):
        match_score = 1 - cv2.matchShapes(contours1[i], contours2[i], cv2.CONTOURS_MATCH_I2, 0)
        shape_score += max(0, match_score)

    # 综合得分（SSIM 40% + 形状匹配60%）
    return 0.4 * ssim_score + 0.6 * (shape_score / min_len)


def get_main_contours(mask):
    """获取主要轮廓（按面积排序前5）"""
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return sorted(contours, key=cv2.contourArea, reverse=True)[:5]


# 测试图片对（保持原列表）
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

# 测试流程（保持原流程）
for pair in image_pairs:
    similarity, angle, mask1, mask2, rotated = calculate_similarity(*pair)
    print(f"相似度: {similarity:.2f}% | 最佳角度: {angle}°")

    # 可视化部分保持不变...