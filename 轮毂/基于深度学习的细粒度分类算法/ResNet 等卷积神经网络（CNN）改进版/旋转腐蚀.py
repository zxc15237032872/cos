import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import cv2
from sklearn.cluster import KMeans

plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置为支持中文的字体，如黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题


def convert_to_square(image, size=70):
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


def light_compensation(image):
    """光照补偿函数"""
    img_array = np.array(image.convert('L'))
    height, width = img_array.shape[:2]
    center_x, center_y = width // 2, height // 2

    angle_bins = 360
    angle_avg = np.zeros(angle_bins)

    for r in range(1, min(center_x, center_y)):
        for angle in range(angle_bins):
            x = int(center_x + r * np.cos(np.radians(angle)))
            y = int(center_y + r * np.sin(np.radians(angle)))
            if 0 <= x < width and 0 <= y < height:
                angle_avg[angle] += img_array[y, x]

    angle_avg /= min(center_x, center_y) - 1

    compensation_map = np.zeros_like(img_array)
    for r in range(height):
        for c in range(width):
            dx = c - center_x
            dy = r - center_y
            angle = np.arctan2(dy, dx) * 180 / np.pi
            if angle < 0:
                angle += 360
            angle_index = int(angle)
            compensation_map[r, c] = angle_avg[angle_index]

    mean_illumination = np.mean(compensation_map)
    corrected_image = img_array * (mean_illumination / compensation_map)
    corrected_image = np.uint8(np.clip(corrected_image, 0, 255))
    return Image.fromarray(corrected_image)


def generate_mask(image):
    """生成mask函数"""
    img_array = np.array(image.convert('L'))
    pixels = img_array.reshape((-1, 1)).astype(np.float32)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 10, 1.0)
    K = 2
    _, labels, centers = cv2.kmeans(pixels, K, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    hub_class = np.argmax(centers)
    mask = (labels == hub_class).reshape(img_array.shape).astype(np.uint8) * 255
    return Image.fromarray(mask)


if __name__ == "__main__":
    image_path1 = "005A.png"  # 替换为实际图片路径
    image_path2 = "005B.png"  # 替换为实际图片路径
    square_image1, square_image2, best_rotated_image2, best_angle = find_best_rotation_angle(image_path1, image_path2)

    # 对旋转后的图像进行腐蚀操作
    kernel = np.ones((1, 1), np.uint8)
    rotated_image2_array = np.array(best_rotated_image2)
    eroded_image = cv2.erode(rotated_image2_array, kernel, iterations=1)
    eroded_image = Image.fromarray(eroded_image)

    # 进行光照补偿
    compensated_image = light_compensation(eroded_image)

    # 进行膨胀操作
    dilated_image_array = np.array(compensated_image)
    dilated_image = cv2.dilate(dilated_image_array, kernel, iterations=1)
    dilated_image = Image.fromarray(dilated_image)

    # 生成mask
    mask = generate_mask(dilated_image)

    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(dilated_image, cmap='gray')
    plt.title('处理后的图像')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.imshow(mask, cmap='gray')
    plt.title('生成的mask')
    plt.axis('off')

    plt.show()
    mask.save("generated_mask.png")  # 保存生成的mask