import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib

# 解决中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 显示图像
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
matplotlib.use('TkAgg')

def equalize_sectors(image, num_sectors):
    # 获取图像的高度和宽度
    height, width = image.shape[:2]
    # 计算图像的中心位置
    center_x, center_y = width // 2, height // 2
    # 计算每个扇区的角度
    sector_angle = 360 / num_sectors

    # 初始化一个列表来存储每个扇区的像素值
    sectors = []

    # 遍历每个扇区
    for i in range(num_sectors):
        # 计算当前扇区的起始角度和结束角度
        start_angle = i * sector_angle
        end_angle = (i + 1) * sector_angle

        # 创建一个掩码，用于提取当前扇区的像素
        mask = np.zeros((height, width), dtype=np.uint8)
        cv2.ellipse(mask, (center_x, center_y), (width // 2, height // 2), 0, start_angle, end_angle, 255, -1)

        # 应用掩码提取当前扇区的像素
        sector = cv2.bitwise_and(image, image, mask=mask)
        sectors.append(sector)

    # 选择第一个扇区作为参考扇区
    reference_sector = sectors[0]
    # 计算参考扇区的平均亮度
    reference_mean = np.mean(reference_sector[reference_sector > 0])

    # 调整其他扇区的亮度
    equalized_sectors = []
    for sector in sectors:
        # 计算当前扇区的平均亮度
        sector_mean = np.mean(sector[sector > 0])
        if sector_mean > 0:
            # 计算亮度调整因子
            factor = reference_mean / sector_mean
            # 对当前扇区的像素值进行调整
            adjusted_sector = (sector * factor).astype(np.uint8)
            equalized_sectors.append(adjusted_sector)
        else:
            equalized_sectors.append(sector)

    # 合并所有调整后的扇区
    equalized_image = np.zeros((height, width), dtype=np.uint8)
    for sector in equalized_sectors:
        equalized_image = cv2.bitwise_or(equalized_image, sector)

    return equalized_image

# 读取图像
image = cv2.imread('005A.png', cv2.IMREAD_GRAYSCALE)

# 检查图像是否成功读取
if image is None:
    print("无法读取图像，请检查图像路径是否正确。")
else:
    # 设置扇区数量
    num_sectors = 40  # 这里将扇区数量设置为 12
    # 进行扇区均衡化处理
    equalized_image = equalize_sectors(image, num_sectors)

    # 显示原始图像和均衡化后的图像
    plt.figure(figsize=(12, 6))
    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title('原始图像')
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(equalized_image, cmap='gray')
    plt.title('扇区均衡化后的图像')
    plt.axis('off')
    plt.show()

    # 保存均衡化后的图像
    cv2.imwrite('equalized_sectors_005A.png', equalized_image)
    print("扇区均衡化后的图像已保存为 equalized_sectors_005A.png")