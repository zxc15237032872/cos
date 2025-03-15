import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import cv2
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def convert_to_square(image, size=64):
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
    # 将 OpenCV 图像转换回 PIL 格式
    rotated_image = Image.fromarray(rotated_img_array)
    return rotated_image

def find_contours(image):
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    _, thresh = cv2.threshold(gray, 100, 240, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
    return contours

def match_contours(contours1, contours2):
    if len(contours1) == 0 or len(contours2) == 0:
        return 0
    # 取最大的轮廓进行匹配
    contour1 = max(contours1, key=cv2.contourArea)
    contour2 = max(contours2, key=cv2.contourArea)
    return cv2.matchShapes(contour1, contour2, cv2.CONTOURS_MATCH_I1, 0)

def find_best_rotation_angle(image_path1, image_path2):
    image1 = Image.open(image_path1).convert('RGB')
    image2 = Image.open(image_path2).convert('RGB')

    # 转换为正圆（内切于正方形）
    square_image1 = convert_to_square(image1)
    square_image2 = convert_to_square(image2)

    contours1 = find_contours(square_image1)

    min_difference = float('inf')
    best_angle = 0
    best_rotated_image2 = None

    # 在 0 到 100 度范围内旋转
    for angle in range(1, 100):
        rotated_image2 = rotate_image_opencv(square_image2, angle)
        contours2 = find_contours(rotated_image2)
        difference = match_contours(contours1, contours2)

        if difference < min_difference:
            min_difference = difference
            best_angle = angle
            best_rotated_image2 = rotated_image2

    print(f"图片 {image_path1} 和 {image_path2} 的最佳旋转角度为: {best_angle} 度")

    return square_image1, square_image2, best_rotated_image2, best_angle

def draw_contours_on_image(image):
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array
    _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(img_array, contours, -1, (0, 255, 0), 1)
    return Image.fromarray(img_array)

# 输入图片路径
image_path1 = '004A.png'
image_path2 = '004A1.png'

# 找到最佳旋转角度并获取处理后的图片
square_image1, square_image2, best_rotated_image2, best_angle = find_best_rotation_angle(image_path1, image_path2)

# 绘制轮廓
square_image1_with_contours = draw_contours_on_image(square_image1)
square_image2_with_contours = draw_contours_on_image(square_image2)
best_rotated_image2_with_contours = draw_contours_on_image(best_rotated_image2)

# 显示图片
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
axes[0].imshow(np.array(square_image1_with_contours))
axes[0].set_title(image_path1)
axes[1].imshow(np.array(square_image2_with_contours))
axes[1].set_title(image_path2)
axes[2].imshow(np.array(best_rotated_image2_with_contours))
axes[2].set_title(f"{image_path2} 旋转 {best_angle} 度后")

for ax in axes:
    ax.axis('off')

plt.show()