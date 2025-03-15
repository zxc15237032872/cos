import cv2
import numpy as np

# 读取轮毂掩码图像
mask = cv2.imread('005A_mask.png', 0)
# 二值化处理（如果图像不是标准二值图）
_, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
# 查找轮廓
contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

spoke_count = 0
for contour in contours:
    # 计算Hu矩
    moments = cv2.moments(contour)
    print(moments)
    hu_moments = cv2.HuMoments(moments).flatten()
    print(hu_moments)
    # 简单设定基于Hu矩的特征阈值筛选辐条轮廓，需根据实际调整
    if np.all(np.abs(hu_moments) > 1e-6):
        spoke_count += 1

print(f"辐条数: {spoke_count}")