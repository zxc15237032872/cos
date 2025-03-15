import cv2
import numpy as np

# 读取图像
image1 = cv2.imread('005A.png', cv2.IMREAD_UNCHANGED)
image2 = cv2.imread('005B.png', cv2.IMREAD_UNCHANGED)
image3 = cv2.imread('006A.png', cv2.IMREAD_UNCHANGED)

# 调整图像尺寸为100x100
image1 = cv2.resize(image1, (100, 100))
image2 = cv2.resize(image2, (100, 100))
image3 = cv2.resize(image3, (100, 100))

# 灰度化处理
gray1 = cv2.cvtColor(image1, cv2.COLOR_BGRA2GRAY)
gray2 = cv2.cvtColor(image2, cv2.COLOR_BGRA2GRAY)
gray3 = cv2.cvtColor(image3, cv2.COLOR_BGRA2GRAY)

# 中值滤波去除噪声
gray1 = cv2.medianBlur(gray1, 5)
gray2 = cv2.medianBlur(gray2, 5)
gray3 = cv2.medianBlur(gray3, 5)

# Canny边缘检测
edges1 = cv2.Canny(gray1, 50, 150)
edges2 = cv2.Canny(gray2, 50, 150)
edges3 = cv2.Canny(gray3, 50, 150)

# Hough圆检测
circles1 = cv2.HoughCircles(edges1, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                            param1=50, param2=30, minRadius=0, maxRadius=0)
circles2 = cv2.HoughCircles(edges2, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                            param1=50, param2=30, minRadius=0, maxRadius=0)
circles3 = cv2.HoughCircles(edges3, cv2.HOUGH_GRADIENT, dp=1, minDist=20,
                            param1=50, param2=30, minRadius=0, maxRadius=0)

# 几何验证
if circles1 is not None:
    circles1 = np.round(circles1[0, :]).astype("int")
    for (x, y, r) in circles1:
        cv2.circle(edges1, (x, y), r, (0, 255, 0), 2)

if circles2 is not None:
    circles2 = np.round(circles2[0, :]).astype("int")
    for (x, y, r) in circles2:
        cv2.circle(edges2, (x, y), r, (0, 255, 0), 2)

if circles3 is not None:
    circles3 = np.round(circles3[0, :]).astype("int")
    for (x, y, r) in circles3:
        cv2.circle(edges3, (x, y), r, (0, 255, 0), 2)

# 形状匹配
# 这里可以根据具体需求实现形状匹配算法

# 生成mask
mask1 = np.zeros_like(gray1)
mask2 = np.zeros_like(gray2)
mask3 = np.zeros_like(gray3)

# 根据检测到的圆形轮廓和形状匹配结果填充mask
# 这里可以根据具体情况进行修改

# 显示结果
cv2.imshow('Image 1', image1)
cv2.imshow('Mask 1', mask1)
cv2.imshow('Image 2', image2)
cv2.imshow('Mask 2', mask2)
cv2.imshow('Image 3', image3)
cv2.imshow('Mask 3', mask3)
cv2.waitKey(0)
cv2.destroyAllWindows()