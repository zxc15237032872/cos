import cv2
import numpy as np

# 读取图像
img1 = cv2.imread(r"image/005B.png", 0)
img2 = cv2.imread(r'image/轮毂001B1.png', 0)

# 创建 ORB 特征检测器
orb = cv2.ORB_create()

# 检测关键点和描述符
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# 创建 BFMatcher 对象
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)

# 匹配描述符
matches = bf.match(des1, des2)

# 按距离排序
matches = sorted(matches, key=lambda x: x.distance)

# 提取匹配点
src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

# 计算变换矩阵
M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

# 应用变换矩阵
h, w = img1.shape
aligned_img = cv2.warpPerspective(img1, M, (w, h))

# 显示结果
cv2.imshow('Aligned Image', aligned_img)
cv2.waitKey(0)
cv2.destroyAllWindows()