import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageFilter
from skimage.metrics import structural_similarity as ssim
import cv2
from sklearn.cluster import KMeans

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置为支持中文的字体，如黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

from PIL import Image, ImageFilter, ImageEnhance


# def preprocess_image(image):
#     # 高斯模糊，适当调整半径，这里设为 0.5 使模糊程度降低，更清晰
#     blurred_image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
#     # 多次锐化以增强锐化效果
#     sharpened_image = blurred_image
#     for _ in range(3):  # 进行 3 次锐化
#         sharpened_image = sharpened_image.filter(ImageFilter.SHARPEN)
#
#     # 增加明暗对比
#     contrast_enhancer = ImageEnhance.Contrast(sharpened_image)
#     # 调整对比度，这里设为 1.5 可以根据实际情况调整
#     contrast_factor = 1.5
#     high_contrast_image = contrast_enhancer.enhance(contrast_factor)
#
#     return high_contrast_image
#
#
# from PIL import Image, ImageFilter, ImageEnhance
#
#
# def preprocess_image(image):
#     # 高斯模糊，适当调整半径，这里设为 0.5 使模糊程度降低，更清晰
#     blurred_image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
#     # 多次锐化以增强锐化效果
#     sharpened_image = blurred_image
#     for _ in range(3):  # 进行 3 次锐化
#         sharpened_image = sharpened_image.filter(ImageFilter.SHARPEN)
#
#     # 增加明暗对比
#     contrast_enhancer = ImageEnhance.Contrast(sharpened_image)
#     # 调整对比度，这里设为 1.5 可以根据实际情况调整
#     contrast_factor = 1.5
#     high_contrast_image = contrast_enhancer.enhance(contrast_factor)
#
#     # 增加饱和度
#     color_enhancer = ImageEnhance.Color(high_contrast_image)
#     # 调整饱和度，这里设为 1.5 可以根据实际情况调整
#     saturation_factor = 1.5
#     high_saturation_image = color_enhancer.enhance(saturation_factor)
#
#     return high_saturation_image

# def preprocess_image(image):
#     # 高斯模糊，适当调整半径，这里设为 0.5 使模糊程度降低，更清晰
#     blurred_image = image.filter(ImageFilter.GaussianBlur(radius=0.5))
#     # 多次锐化以增强锐化效果
#     sharpened_image = blurred_image
#     for _ in range(3):  # 进行 3 次锐化
#         sharpened_image = sharpened_image.filter(ImageFilter.SHARPEN)
#
#     # 增加明暗对比
#     contrast_enhancer = ImageEnhance.Contrast(sharpened_image)
#     # 调整对比度，这里设为 1.5 可以根据实际情况调整
#     contrast_factor = 1.5
#     high_contrast_image = contrast_enhancer.enhance(contrast_factor)
#
#     # 增加饱和度
#     color_enhancer = ImageEnhance.Color(high_contrast_image)
#     # 调整饱和度，这里设为 1.5 可以根据实际情况调整
#     saturation_factor = 1.5
#     high_saturation_image = color_enhancer.enhance(saturation_factor)
#
#     return high_saturation_image

from PIL import Image, ImageFilter, ImageEnhance
import cv2
import numpy as np

# def preprocess_image(image):
#     # 将 PIL 图像转换为 OpenCV 格式
#     img = np.array(image)
#
#     # 高斯模糊，适当调整半径，这里设为 0.5 使模糊程度降低，更清晰
#     blurred_image = cv2.GaussianBlur(img, (3, 3), 0.5)
#
#     # 自适应直方图均衡化（CLAHE）
#     lab = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     cl = clahe.apply(l)
#     limg = cv2.merge((cl, a, b))
#     clahe_image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
#
#     # 伽马校正
#     gamma = 0.7  # 可以根据实际情况调整伽马值
#     inv_gamma = 1.0 / gamma
#     table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
#     gamma_corrected_image = cv2.LUT(clahe_image, table)
#
#     # 将 OpenCV 图像转换回 PIL 格式
#     enhanced_image = Image.fromarray(gamma_corrected_image)
#
#     # 多次锐化以增强锐化效果
#     sharpened_image = enhanced_image
#     for _ in range(0):  # 进行 3 次锐化
#         sharpened_image = sharpened_image.filter(ImageFilter.SHARPEN)
#
#     # 增加明暗对比
#     contrast_enhancer = ImageEnhance.Contrast(sharpened_image)
#     # 调整对比度，这里设为 1.5 可以根据实际情况调整
#     contrast_factor = 2
#     high_contrast_image = contrast_enhancer.enhance(contrast_factor)
#
#     # 增加饱和度
#     color_enhancer = ImageEnhance.Color(high_contrast_image)
#     # 调整饱和度，这里设为 1.5 可以根据实际情况调整
#     saturation_factor = 3
#     high_saturation_image = color_enhancer.enhance(saturation_factor)
#
#     return high_saturation_image



from PIL import Image, ImageFilter, ImageEnhance
import cv2
import numpy as np

# def preprocess_image(image):
#     # 将 PIL 图像转换为 OpenCV 格式
#     img = np.array(image)
#
#     # 高斯模糊，适当调整半径，这里设为 0.5 使模糊程度降低，更清晰
#     blurred_image = cv2.GaussianBlur(img, (3, 3), 0.5)
#
#     # 自适应直方图均衡化（CLAHE）
#     lab = cv2.cvtColor(blurred_image, cv2.COLOR_RGB2LAB)
#     l, a, b = cv2.split(lab)
#     clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
#     cl = clahe.apply(l)
#     limg = cv2.merge((cl, a, b))
#     clahe_image = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
#
#     # 伽马校正
#     gamma = 0.7  # 可以根据实际情况调整伽马值
#     inv_gamma = 1.0 / gamma
#     table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in np.arange(0, 256)]).astype("uint8")
#     gamma_corrected_image = cv2.LUT(clahe_image, table)
#
#     # 引导滤波结合局部均值调整亮度
#     gray = cv2.cvtColor(gamma_corrected_image, cv2.COLOR_RGB2GRAY)
#     guide = cv2.ximgproc.guidedFilter(gray, gray, radius=10, eps=100)
#     local_mean = cv2.boxFilter(gray, -1, (11, 11))  # 计算局部均值
#     diff = local_mean - gray
#     adjust_mask = diff > 10  # 只调整亮度差异大于 10 的区域
#     enhanced_gray = gray.copy()
#     enhanced_gray[adjust_mask] += diff[adjust_mask]
#     enhanced_gray = np.clip(enhanced_gray, 0, 255).astype(np.uint8)
#     enhanced_image = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
#
#     # 将 OpenCV 图像转换回 PIL 格式
#     enhanced_image = Image.fromarray(enhanced_image)
#
#     # 多次锐化以增强锐化效果
#     sharpened_image = enhanced_image
#     for _ in range(3):  # 进行 3 次锐化
#         sharpened_image = sharpened_image.filter(ImageFilter.SHARPEN)
#
#     # 增加明暗对比
#     contrast_enhancer = ImageEnhance.Contrast(sharpened_image)
#     # 调整对比度，这里设为 2 可以根据实际情况调整
#     contrast_factor = 2
#     high_contrast_image = contrast_enhancer.enhance(contrast_factor)
#
#     # 增加饱和度
#     color_enhancer = ImageEnhance.Color(high_contrast_image)
#     # 调整饱和度，这里设为 3 可以根据实际情况调整
#     saturation_factor = 3
#     high_saturation_image = color_enhancer.enhance(saturation_factor)
#
#     return high_saturation_image
import cv2
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance

def histogram_matching(source, template):
    """直方图匹配"""
    oldshape = source.shape
    source = source.ravel()
    template = template.ravel()

    s_values, bin_idx, s_counts = np.unique(source, return_inverse=True, return_counts=True)
    t_values, t_counts = np.unique(template, return_counts=True)

    s_quantiles = np.cumsum(s_counts).astype(np.float64)
    s_quantiles /= s_quantiles[-1]
    t_quantiles = np.cumsum(t_counts).astype(np.float64)
    t_quantiles /= t_quantiles[-1]

    interp_t_values = np.interp(s_quantiles, t_quantiles, t_values)
    return interp_t_values[bin_idx].reshape(oldshape)

def preprocess_image(image):
    img = np.array(image)
    # 假设参考图像是一个均匀分布的图像
    reference = np.random.randint(0, 256, size=img.shape, dtype=np.uint8)
    enhanced_img = histogram_matching(img, reference).astype(np.uint8)

    # 转换为 PIL 图像
    enhanced_image = Image.fromarray(enhanced_img)

    # 多次锐化以增强锐化效果
    sharpened_image = enhanced_image
    for _ in range(3):
        sharpened_image = sharpened_image.filter(ImageFilter.SHARPEN)

    # 增加明暗对比
    contrast_enhancer = ImageEnhance.Contrast(sharpened_image)
    contrast_factor = 2
    high_contrast_image = contrast_enhancer.enhance(contrast_factor)

    # 增加饱和度
    color_enhancer = ImageEnhance.Color(high_contrast_image)
    saturation_factor = 3
    high_saturation_image = color_enhancer.enhance(saturation_factor)

    return high_saturation_image
def convert_to_square(image, size=70):
    # 预处理图像
    image = preprocess_image(image)
    # 将图像转换为灰度图以便处理
    gray_image = image.convert('L')
    # 将图像转换为 numpy 数组
    img_array = np.array(gray_image)

    # 找到非零像素的坐标
    rows, cols = np.nonzero(img_array)
    if len(rows) == 0 or len(cols) == 0:
        return Image.new("RGB", (size, size), (0, 0, 0))

    # 计算椭圆的边界
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    # 计算椭圆的长短轴长度
    major_axis = max(max_row - min_row, max_col - min_col)
    minor_axis = min(max_row - min_row, max_col - min_col)

    # 计算缩放比例
    scale_ratio = major_axis / minor_axis if minor_axis != 0 else 1

    # 根据长短轴方向进行缩放
    if max_row - min_row > max_col - min_col:
        new_width = int(image.width * scale_ratio)
        resized_image = image.resize((new_width, image.height))
    else:
        new_height = int(image.height * scale_ratio)
        resized_image = image.resize((image.width, new_height))

    # 重新计算灰度图和非零像素坐标
    gray_resized = resized_image.convert('L')
    resized_array = np.array(gray_resized)
    rows, cols = np.nonzero(resized_array)

    # 再次计算椭圆的边界
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)

    # 裁剪出包含椭圆的正方形区域
    left = min_col
    top = min_row
    right = max_col
    bottom = max_row
    cropped_image = resized_image.crop((left, top, right, bottom))

    # 调整图像大小为指定尺寸
    final_image = cropped_image.resize((size, size))

    # 只保留圆内的区域
    center = (size // 2, size // 2)
    radius = size // 2
    final_array = np.array(final_image)
    for y in range(size):
        for x in range(size):
            if (x - center[0]) ** 2 + (y - center[1]) ** 2 > radius ** 2:
                final_array[y, x] = [0, 0, 0]
    final_image = Image.fromarray(final_array)

    return final_image


def rotate_image_opencv(image, angle):
    # 将 PIL 图像转换为 OpenCV 格式
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    # 计算旋转中心
    center = (width // 2, height // 2)
    # 获取旋转矩阵
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    # 进行旋转操作
    rotated_img_array = cv2.warpAffine(img_array, rotation_matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)

    # 只保留圆内的区域
    size = min(width, height)
    center = (width // 2, height // 2)
    radius = size // 2
    for y in range(height):
        for x in range(width):
            if (x - center[0]) ** 2 + (y - center[1]) ** 2 > radius ** 2:
                rotated_img_array[y, x] = [0, 0, 0]

    # 将 OpenCV 图像转换回 PIL 格式
    rotated_image = Image.fromarray(rotated_img_array)
    return rotated_image


def calculate_ssim(image1, image2):
    img1_array = np.array(image1.convert('L'))
    img2_array = np.array(image2.convert('L'))
    return ssim(img1_array, img2_array)


def find_best_rotation_angle(image_path1, image_path2):
    image1 = Image.open(image_path1).convert('RGB')
    image2 = Image.open(image_path2).convert('RGB')

    # 转换为正圆（内切于正方形）
    square_image1 = convert_to_square(image1)
    square_image2 = convert_to_square(image2)

    max_similarity = -1
    best_angle = 0
    best_rotated_image2 = None

    # 细化旋转角度搜索，步长为 1 度
    for angle in np.arange(0, 100, 1):
        rotated_image2 = rotate_image_opencv(square_image2, angle)
        similarity = calculate_ssim(square_image1, rotated_image2)

        if similarity > max_similarity:
            max_similarity = similarity
            best_angle = angle
            best_rotated_image2 = rotated_image2

    print(f"图片 {image_path1} 和 {image_path2} 的最佳旋转角度为: {best_angle} 度")

    return square_image1, square_image2, best_rotated_image2, best_angle


def find_two_main_colors(image):
    """
    使用 K-Means 聚类找到图像中的两种主要颜色，将接近银灰色的设为红色，黑色保持黑色
    """
    img = np.array(image)
    pixels = img.reshape(-1, 3)
    kmeans = KMeans(n_clusters=2, random_state=0).fit(pixels)
    labels = kmeans.labels_
    center1, center2 = kmeans.cluster_centers_

    def is_close_to_silver_gray(color):
        # 银灰色的大致范围
        lower_bound = np.array([100, 100, 100])
        upper_bound = np.array([220, 220, 220])
        return np.all(color >= lower_bound) and np.all(color <= upper_bound)

    if is_close_to_silver_gray(center1):
        silver_gray_center = center1
        black_center = center2
    elif is_close_to_silver_gray(center2):
        silver_gray_center = center2
        black_center = center1
    else:
        # 如果两个聚类中心都不接近银灰色，选择亮度较高的作为银灰色
        if np.mean(center1) > np.mean(center2):
            silver_gray_center = center1
            black_center = center2
        else:
            silver_gray_center = center2
            black_center = center1

    new_image = np.zeros_like(img)
    for i, label in enumerate(labels):
        if np.array_equal(kmeans.cluster_centers_[label], silver_gray_center):
            new_image[i // img.shape[1], i % img.shape[1]] = [255, 0, 0]  # 接近银灰色的设为红色
        else:
            new_image[i // img.shape[1], i % img.shape[1]] = [0, 0, 0]  # 黑色

    return Image.fromarray(new_image.astype(np.uint8))


def extract_features(image):
    img = np.array(image).flatten()
    return img


def similarity(feature1, feature2):
    """
    计算参照物红色区域与旋转后红色区域的交集与它们的并集的比值
    """
    size = int(np.sqrt(len(feature1) / 3))
    feature1 = feature1.reshape(size, size, 3)
    feature2 = feature2.reshape(size, size, 3)

    # 找到红色区域
    red_area1 = np.all(feature1 == [255, 0, 0], axis=2)
    red_area2 = np.all(feature2 == [255, 0, 0], axis=2)

    # 计算交集
    intersection = np.logical_and(red_area1, red_area2).sum()

    # 计算并集
    union = np.logical_or(red_area1, red_area2).sum()

    if union == 0:
        return 0
    return intersection / union


image_pairs = [

    ('004A.png', '004A1.png'),
    ('005A.png', '005B.png'),

]

for pair in image_pairs:
    image_path1, image_path2 = pair
    # 找到最佳旋转角度并获取处理后的图片
    square_image1, square_image2, best_rotated_image2, best_angle = find_best_rotation_angle(image_path1, image_path2)

    # 找到两种主要颜色并赋予黑色和红色
    colored_image1 = find_two_main_colors(square_image1)
    colored_image2 = find_two_main_colors(square_image2)
    colored_rotated_image2 = find_two_main_colors(best_rotated_image2)

    # 提取特征
    feature1 = extract_features(colored_image1)
    feature2 = extract_features(colored_image2)
    feature2_rotated = extract_features(colored_rotated_image2)

    # 计算相似度
    similarity_before_rotation = similarity(feature1, feature2)
    similarity_after_rotation = similarity(feature1, feature2_rotated)

    print(f"旋转前 {image_path1} 和 {image_path2} 的相似度: {similarity_before_rotation}")
    print(f"旋转 {best_angle} 度后 {image_path1} 和 {image_path2} 的相似度: {similarity_after_rotation}")

    # 显示图片，减小图片大小
    fig, axes = plt.subplots(3, 3, figsize=(8, 8))
    axes[0, 0].imshow(np.array(square_image1))
    axes[0, 0].set_title(f"{image_path1} 锐化后")
    axes[0, 1].imshow(np.array(square_image2))
    axes[0, 1].set_title(f"{image_path2} 锐化后")
    axes[0, 2].imshow(np.array(best_rotated_image2))
    axes[0, 2].set_title(f"{image_path2} 旋转 {best_angle} 度锐化后")

    axes[1, 0].imshow(np.array(colored_image1))
    axes[1, 0].set_title(f"{image_path1} 颜色处理后")
    axes[1, 1].imshow(np.array(colored_image2))
    axes[1, 1].set_title(f"{image_path2} 颜色处理后")
    axes[1, 2].imshow(np.array(colored_rotated_image2))
    axes[1, 2].set_title(f"{image_path2} 旋转 {best_angle} 度颜色处理后")

    axes[2, 0].imshow(np.array(Image.fromarray(feature1.reshape(colored_image1.size[1], colored_image1.size[0], 3))), cmap='gray')
    axes[2, 0].set_title(f"{image_path1} 特征可视化")
    axes[2, 1].imshow(np.array(Image.fromarray(feature2.reshape(colored_image2.size[1], colored_image2.size[0], 3))), cmap='gray')
    axes[2, 1].set_title(f"{image_path2} 特征可视化")
    axes[2, 2].imshow(np.array(Image.fromarray(feature2_rotated.reshape(colored_rotated_image2.size[1], colored_rotated_image2.size[0], 3))), cmap='gray')
    axes[2, 2].set_title(f"{image_path2} 旋转 {best_angle} 度特征可视化")

    for ax in axes.flat:
        ax.axis('off')

    plt.tight_layout()
    plt.show()