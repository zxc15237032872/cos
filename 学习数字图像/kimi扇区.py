import cv2
import numpy as np
import cv2
import numpy as np

def rotate_image(image, angle, scale=1.0):
    """
    旋转图像
    :param image: 输入图像
    :param angle: 旋转角度（逆时针为正）
    :param scale: 缩放比例（默认为1.0）
    :return: 旋转后的图像
    """
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)  # 图像中心作为旋转中心
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated_image = cv2.warpAffine(image, rotation_matrix, (w, h))
    return rotated_image

def is_in_sector(x, y, cx, cy, start_angle, end_angle, r_min, r_max):
    """
    判断点是否在扇区内
    :param x, y: 像素点坐标
    :param cx, cy: 扇区中心坐标
    :param start_angle, end_angle: 扇区的起始和终止角度
    :param r_min, r_max: 扇区的最小和最大半径
    :return: 布尔值
    """
    dx = x - cx
    dy = y - cy
    r = np.sqrt(dx**2 + dy**2)
    theta = np.degrees(np.arctan2(dy, dx)) % 360  # 转换为角度并取模

    if r_min <= r <= r_max and start_angle <= theta <= end_angle:
        return True
    return False

def calculate_sector(image, cx, cy, start_angle, end_angle, r_min, r_max):
    """
    计算扇区内的像素
    :param image: 输入图像
    :param cx, cy: 扇区中心坐标
    :param start_angle, end_angle: 扇区的起始和终止角度
    :param r_min, r_max: 扇区的最小和最大半径
    :return: 扇区掩码
    """
    h, w = image.shape[:2]
    sector_mask = np.zeros((h, w), dtype=np.uint8)

    for y in range(h):
        for x in range(w):
            if is_in_sector(x, y, cx, cy, start_angle, end_angle, r_min, r_max):
                sector_mask[y, x] = 255
    return sector_mask
# 读取图像
image = cv2.imread('005A.png')

# 旋转图像
angle = 45  # 旋转角度
rotated_image = rotate_image(image, angle)

# 计算扇区
cx, cy = rotated_image.shape[1] // 2, rotated_image.shape[0] // 2  # 扇区中心
start_angle = 0  # 扇区起始角度
end_angle = 90  # 扇区终止角度
r_min = 50  # 扇区最小半径
r_max = 150  # 扇区最大半径

sector_mask = calculate_sector(rotated_image, cx, cy, start_angle, end_angle, r_min, r_max)

# 显示结果
cv2.imshow('Rotated Image', rotated_image)
cv2.imshow('Sector Mask', sector_mask)
cv2.waitKey(0)
cv2.destroyAllWindows()