import cv2
import numpy as np


# 读取图像
image = cv2.imread('006A.png')
if image is None:
    raise FileNotFoundError('无法读取图像006A.png')

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# 使用霍夫圆变换检测圆
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1, 20,
                           param1=50, param2=30, minRadius=0, maxRadius=0)

if circles is not None:
    circles = np.uint16(np.around(circles))
    for i in circles[0, :]:
        center = (i[0], i[1])
        radius = i[2]
        # param1：霍夫梯度法中Canny边缘检测的高阈值，低阈值是高阈值的一半。
        # 这里设为50，较高的值会减少边缘检测的数量，有助于排除一些噪声边缘，但如果过高可能会遗漏轮毂的边缘。
        # param2：累加器阈值。只有累加器值高于此阈值的圆才会被检测出来。
        # 设为30，较小的值会检测出更多的圆，包括一些可能是噪声的假圆；较大的值则只会检测出更显著的圆。
        # minRadius：检测到的圆的最小半径。设为0，表示不限制最小半径，实际应用中可根据轮毂大小调整。
        # maxRadius：检测到的圆的最大半径。设为0，表示不限制最大半径，同样可根据实际轮毂尺寸调整。

        # 创建一个空白图像，大小与原图像相同
        blank_image = np.zeros_like(image)
        # 在空白图像上绘制标准圆
        cv2.circle(blank_image, center, radius, (255, 255, 255), thickness=-1)
        # 使用按位与操作将原图像中的轮毂部分提取出来并规整为标准圆
        result = cv2.bitwise_and(image, blank_image)
        # 显示结果图像
        cv2.imshow('result', result)
        cv2.waitKey(0)

        break
else:
    print('未检测到圆')
