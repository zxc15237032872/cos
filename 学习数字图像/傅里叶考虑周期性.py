import cv2
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import find_peaks

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

    # 转换为 PIL 图像
    pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    # 将图像转换为灰度图以便处理
    gray_image = pil_image.convert('L')
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
        new_width = int(pil_image.width * scale_ratio)
        resized_image = pil_image.resize((new_width, pil_image.height))
    else:
        new_height = int(pil_image.height * scale_ratio)
        resized_image = pil_image.resize((pil_image.width, new_height))

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
image_path = '005B.png'
final_image, center, radius = convert_to_circle(image_path, size=64)

print(f"圆心坐标: {center}")
print(f"半径: {radius}")

# 将 PIL 图像转换为 numpy 数组
array_image = np.array(final_image)

# 转换为灰度图像
gray_image = cv2.cvtColor(array_image, cv2.COLOR_RGB2GRAY)
# 高斯模糊


# 锐化操作：使用拉普拉斯算子
laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
sharpened_image = gray_image - laplacian
sharpened_image = np.uint8(np.clip(sharpened_image, 0, 255))
gaussian_image = cv2.GaussianBlur(sharpened_image, (5, 5), 0)
# 进行二维傅里叶变换
f = np.fft.fft2(gaussian_image)
# 频谱中心化
fshift = np.fft.fftshift(f)
# 计算频谱幅度
magnitude_spectrum = 20 * np.log(np.abs(fshift) + 1e-10)  # 避免 log(0) 警告

# 找到频谱中心
rows, cols = sharpened_image.shape
crow, ccol = rows // 2, cols // 2

# 动态调整环形区域半径
radius_inner = int(radius * 0.2)
radius_outer = int(radius * 0.8)

# 提取频谱的环形区域
mask = np.zeros((rows, cols), np.uint8)
y, x = np.ogrid[:rows, :cols]
mask_area = (x - ccol) ** 2 + (y - crow) ** 2 <= radius_outer ** 2
mask_area &= (x - ccol) ** 2 + (y - crow) ** 2 >= radius_inner ** 2
mask[mask_area] = 1
fshift_masked = fshift * mask
magnitude_spectrum_masked = 20 * np.log(np.abs(fshift_masked) + 1e-10)  # 避免 log(0) 警告

# 计算角度方向上的能量分布
angles = np.linspace(0, 2 * np.pi, 360)
energy_angles = []
for angle in angles:
    x = int(ccol + radius_outer * np.cos(angle))
    y = int(crow + radius_outer * np.sin(angle))
    energy_angles.append(magnitude_spectrum_masked[y, x])

# 找到能量分布的峰值
peaks, _ = find_peaks(energy_angles, height=10, distance=10)

# 计算相邻峰值之间的平均角度间隔
if len(peaks) > 1:
    peak_angles = [angles[i] for i in peaks]
    angle_differences = np.diff(peak_angles)
    average_angle_difference = np.mean(angle_differences)

    # 根据平均角度间隔推测完整的辐条数量
    total_spoke_count = int(2 * np.pi / average_angle_difference)
else:
    total_spoke_count = len(peaks)

print(f"轮毂的辐条数: {total_spoke_count}")

# 显示裁剪后的圆形图像、锐化后的图像和频谱
plt.figure(figsize=(15, 5))
plt.subplot(131)
plt.imshow(final_image)
plt.title('裁剪后的圆形图像')
plt.axis('off')

plt.subplot(132)
plt.imshow(sharpened_image, cmap='gray')
plt.title('锐化后的图像')
plt.axis('off')

plt.subplot(133)
plt.imshow(magnitude_spectrum_masked, cmap='gray')
plt.title('频谱（环形区域）')
plt.axis('off')

plt.show()