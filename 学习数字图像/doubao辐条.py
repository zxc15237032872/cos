import cv2
import numpy as np

# 读取图像
image = cv2.imread('005A_mask.png')

# 确定旋转中心点和旋转角度
center = (200, 200)  # 旋转中心点
angle = 45  # 旋转角度

# 构建旋转矩阵
M = cv2.getRotationMatrix2D(center, angle, 1.0)

# 执行旋转操作
rotated_image = cv2.warpAffine(image, M, (image.shape[1], image.shape[0]))

# 显示旋转后的图像
cv2.imshow('Rotated Image', rotated_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
