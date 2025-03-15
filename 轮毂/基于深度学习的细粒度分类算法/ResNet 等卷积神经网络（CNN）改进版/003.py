import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt



def align_lines(image1, image2):
    # 将图像转换为灰度图
    gray1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)

    # 使用Canny边缘检测
    edges1 = cv2.Canny(gray1, 50, 150)
    edges2 = cv2.Canny(gray2, 50, 150)

    # 使用霍夫直线变换检测直线
    lines1 = cv2.HoughLines(edges1, 1, np.pi / 180, 150)
    lines2 = cv2.HoughLines(edges2, 1, np.pi / 180, 150)

    # 提取第一条直线的角度
    angle1 = np.arctan2(lines1[0][0][1], lines1[0][0][0])

    # 提取第二条直线的角度
    angle2 = np.arctan2(lines2[0][0][1], lines2[0][0][0])

    # 计算角度差
    delta_angle = angle1 - angle2

    # 获取图像的中心
    (h, w) = image2.shape[:2]
    center = (w // 2, h // 2)

    # 计算旋转矩阵
    M = cv2.getRotationMatrix2D(center, np.degrees(delta_angle), 1.0)

    # 对图像进行旋转
    rotated = cv2.warpAffine(image2, M, (w, h))

    return rotated


# 读取图像
image1 = cv2.imread('004A.png')
image2 = cv2.imread('004A1.png')
print(image1.shape)
print(image2.shape)

# 调用函数进行直线对齐
aligned_image = align_lines(image1, image2)

# 在矫正后的图片上绘制直线
gray_aligned = cv2.cvtColor(aligned_image, cv2.COLOR_BGR2GRAY)
edges_aligned = cv2.Canny(gray_aligned, 50, 150)
lines_aligned = cv2.HoughLines(edges_aligned, 1, np.pi / 180, 150)
for line in lines_aligned:
    rho, theta = line[0]
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + 1000 * (-b))
    y1 = int(y0 + 1000 * (a))
    x2 = int(x0 - 1000 * (-b))
    y2 = int(y0 - 1000 * (a))
    cv2.line(aligned_image, (x1, y1), (x2, y2), (0, 0, 255), 2)

# 展示矫正后的图片
cv2.imshow('Aligned and Marked Image', aligned_image)
cv2.waitKey(0)
cv2.destroyAllWindows()