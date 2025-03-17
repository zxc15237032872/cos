import cv2
import numpy as np

# 读取图像
image = cv2.imread('009A.png')

# 将彩色图像转换为灰度图像
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 使用阈值化将灰度图像转换为二值图像
_, binary_image = cv2.threshold(gray_image, 127, 255, cv2.THRESH_BINARY)

# 寻找图像中的轮廓
contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

# 创建一个与原图像大小相同的黑色图像
result = np.zeros_like(binary_image)

# 显示原始图像和二值图像
cv2.imshow('Original Image', image)
cv2.imshow('Binary Image', binary_image)

# 动态绘制轮廓
for i, contour in enumerate(contours):
    # 复制当前结果图像
    temp_result = result.copy()
    # 绘制当前轮廓
    cv2.drawContours(temp_result, [contour], -1, 255, thickness=cv2.FILLED)
    # 显示绘制当前轮廓后的图像
    cv2.imshow('Processed Image', temp_result)
    # 暂停半秒
    cv2.waitKey(500)

# 最后绘制最大轮廓
max_contour = max(contours, key=cv2.contourArea)
cv2.drawContours(result, [max_contour], -1, 255, thickness=cv2.FILLED)
cv2.imshow('Processed Image', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存处理后的图像
cv2.imwrite('processed_binary_image.jpg', result)