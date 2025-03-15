import cv2
import numpy as np

# 读取图片
image = cv2.imread('005A.png')
if image is None:
    print("无法读取图像，请检查图像路径和文件名。")
else:
    # 调整图片尺寸为100x100
    image = cv2.resize(image, (640, 640))

    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 中值滤波，相比高斯滤波，中值滤波在去除椒盐噪声的同时能更好地保留边缘
    blurred = cv2.medianBlur(gray, 3)

    # 自适应阈值处理，能更好地适应图像不同区域的光照变化
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历轮廓
    for contour in contours:
        # 计算轮廓的面积
        area = cv2.contourArea(contour)
        # 筛选合适面积范围的轮廓，可根据实际情况调整

        if 10 < area < 2000:

            # 计算轮廓的周长
            perimeter = cv2.arcLength(contour, True)
            # 计算轮廓的圆形度（圆形度接近1表示接近圆形）
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            # 设置圆形度的阈值，这里设为0.8，可根据实际情况调整
            if circularity > 0.8:
                # 找到最小外接圆
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                print("圆心坐标：", center)
                radius = int(radius)
                print("半径：", radius)

                # 绘制圆
                cv2.circle(image, center, radius, (0, 255, 0), 1)
                # 标记圆心
                cv2.circle(image, center, 2, (0, 0, 255), -1)

    # 显示结果
    cv2.imshow('Detected Circles', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()