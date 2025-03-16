import cv2
import numpy as np

# 读取二值图像
image = cv2.imread('005A.png', 0)
_, binary = cv2.threshold(image, 50, 255, cv2.THRESH_BINARY)

# 定义结构元素
kernel = np.ones((3, 3), np.uint8)

# 计算形态学梯度
morphological_gradient = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel)

# 显示结果
cv2.imshow('Morphological Gradient', morphological_gradient)
cv2.waitKey(0)
cv2.destroyAllWindows()