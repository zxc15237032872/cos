import cv2
import numpy as np

# 读取二值图像
image = cv2.imread('img_3.png', 0)
image=cv2.resize(image,(80,80))

# 高斯模糊，用于平滑图像并减少噪声
blurred_image = cv2.GaussianBlur(image, (1, 1), 0)

# 闭运算，用于填充小孔和连接断裂的边界
kernel = np.ones((1, 1), np.uint8)
closed_image = cv2.morphologyEx(blurred_image, cv2.MORPH_CLOSE, kernel)

# 查找轮廓
contours, _ = cv2.findContours(closed_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 轮廓近似，用于简化轮廓
approx_contours = []
for contour in contours:
    epsilon = 0.01 * cv2.arcLength(contour, True)
    approx = cv2.approxPolyDP(contour, epsilon, True)
    approx_contours.append(approx)

# 创建一个空白图像用于绘制轮廓
output_image = np.zeros_like(image)

# 重新绘制轮廓
cv2.drawContours(output_image, approx_contours, 5, 255, 2)

# 显示结果
cv2.imshow('Original Image', image)
cv2.imshow('Blurred Image', blurred_image)
cv2.imshow('Closed Image', closed_image)
cv2.imshow('Output Image', output_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
