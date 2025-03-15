import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib

# 解决中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 显示图像
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
matplotlib.use('TkAgg')


def convert_to_circle(image_path, size=64):
    # 读取图像
    image = cv2.imread(image_path)
    if image is None:
        print("无法读取图像，请检查图像路径是否正确。")
        return

    log_transformed = image
    # 平滑滤波（高斯平滑）
    smoothed = cv2.GaussianBlur(log_transformed, (5, 5), 0)
    # 设置亮度阈值为 20%
    threshold_value = int(0.2 * 255)
    _, binary = cv2.threshold(smoothed, threshold_value, 255, cv2.THRESH_BINARY)

    # 转换为 PIL 图像
    binary_pil = Image.fromarray(cv2.cvtColor(binary, cv2.COLOR_BGR2RGB))

    # 将图像转换为灰度图以便处理
    gray_image = binary_pil.convert('L')
    # 将图像转换为 numpy 数组
    img_array = np.array(gray_image)

    # 找到非零像素的坐标
    rows, cols = np.nonzero(img_array)
    if len(rows) == 0 or len(cols) == 0:
        return Image.new("RGB", (size, size), (0, 0, 0)), (0, 0), 0

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
        new_width = int(binary_pil.width * scale_ratio)
        resized_image = binary_pil.resize((new_width, binary_pil.height))
    else:
        new_height = int(binary_pil.height * scale_ratio)
        resized_image = binary_pil.resize((binary_pil.width, new_height))

    # 重新计算灰度图和非零像素坐标
    gray_resized = resized_image.convert('L')
    resized_array = np.array(gray_resized)
    rows, cols = np.nonzero(resized_array)

    # 再次计算椭圆的边界
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)

    # 计算中心
    center_x = (min_col + max_col) // 2
    center_y = (min_row + max_row) // 2
    # 计算半径
    radius = max(max_row - min_row, max_col - min_col) // 2

    # 创建圆形掩码
    mask = Image.new('L', resized_image.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((center_x - radius, center_y - radius, center_x + radius, center_y + radius), fill=255)

    # 应用掩码裁剪图像
    result = Image.composite(resized_image, Image.new("RGB", resized_image.size, (0, 0, 0)), mask)

    # 调整图像大小为指定尺寸
    final_image = result.resize((size, size))

    return final_image, (center_x, center_y), radius


# 调用函数进行转换
image_path = '005A.png'
final_image, center, radius = convert_to_circle(image_path, size=64)

print(f"圆心坐标: {center}")
print(f"半径: {radius}")

# 将 PIL 图像转换为 numpy 数组
binary_array = np.array(final_image.convert('L'))

# 使用中值滤波器进行滤波
filtered_image = cv2.medianBlur(binary_array, 3)

# 显示原始圆形图像和滤波后的图像
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(binary_array, cmap='gray')
plt.title('裁剪后的圆形图像')
plt.axis('off')

plt.subplot(122)
plt.imshow(filtered_image, cmap='gray')
plt.title('中值滤波后的图像')
plt.axis('off')

plt.show()