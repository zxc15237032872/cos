# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
# import matplotlib
#
# # 解决中文显示问题
# matplotlib.rcParams['font.sans-serif'] = ['SimHei']
# matplotlib.rcParams['axes.unicode_minus'] = False
# # 显示图像
# plt.rcParams['font.sans-serif'] = ['SimHei']
# plt.rcParams['axes.unicode_minus'] = False
# matplotlib.use('TkAgg')
# # 读取并预处理图像
# image = cv2.imread('005A.png', 0)
# image = cv2.resize(image, (100, 100))
#
# # 应用CLAHE进行自适应直方图均衡化
# clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
# corrected_image = clahe.apply(image)
# corrected_image = cv2.erode(corrected_image, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))
#
# # 创建一个空白图像用于累加轮廓
# accumulated_image = np.zeros((100, 100), dtype=np.uint8)
#
# # 遍历阈值从10到230
# for threshold in range(90, 250, 10):
#     ret, binary_image = cv2.threshold(corrected_image, threshold, 255, cv2.THRESH_BINARY)
#     contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     cv2.drawContours(accumulated_image, contours, -1, (255), 1)
#
# # 显示累加轮廓的图像
# plt.figure(figsize=(9, 1.5))
# plt.subplot(141)
# plt.imshow(corrected_image, cmap='gray')
# plt.title('自适应直方图均衡化')
#
# plt.subplot(142)
# plt.imshow(accumulated_image, cmap='gray')
# plt.title('累加轮廓')
#
#
# #腐蚀
# kel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
# cv2.erode(accumulated_image,kel , iterations=1)
# plt.subplot(143)
# plt.imshow(accumulated_image, cmap='gray')
# plt.title('累加轮廓')
#
#
# plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 解决中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False
# 显示图像
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
matplotlib.use('TkAgg')

# 读取并预处理图像
image = cv2.imread('005A.png', 0)
image = cv2.resize(image, (100, 100))

# 应用CLAHE进行自适应直方图均衡化
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
corrected_image = clahe.apply(image)
corrected_image = cv2.erode(corrected_image, cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1)))

# 创建一个空白图像用于累加轮廓
accumulated_image = np.zeros((100, 100), dtype=np.uint8)

# 遍历阈值从10到230
for threshold in range(90, 250, 10):
    ret, binary_image = cv2.threshold(corrected_image, threshold, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(binary_image, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(accumulated_image, contours, -1, (255), 1)

# 显示累加轮廓的图像
plt.figure(figsize=(12, 3))
plt.subplot(151)
plt.imshow(corrected_image, cmap='gray')
plt.title('自适应直方图均衡化')

plt.subplot(152)
plt.imshow(accumulated_image, cmap='gray')
plt.title('原始轮廓')

# 准备用于聚类的数据
# 获取轮廓图像中像素值不为0的像素坐标
rows, cols = np.nonzero(accumulated_image)
points = np.column_stack((rows, cols))
points = np.float32(points)

# 定义K-Means算法的终止条件
criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

# 应用K-Means算法，这里假设分为2类
k = 2
_, labels, centers = cv2.kmeans(points, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

# 创建一个空白图像用于显示聚类结果
clustered_image = np.zeros_like(accumulated_image)

# 假设轮毂对应的聚类中心在图像中心附近，选择离图像中心最近的聚类中心作为轮毂所在的类
center_row, center_col = accumulated_image.shape[0] // 2, accumulated_image.shape[1] // 2
distances = np.sqrt((centers[:, 0] - center_row) ** 2 + (centers[:, 1] - center_col) ** 2)
hub_index = np.argmin(distances)

# 将属于轮毂类别的像素点绘制到新图像上
for i in range(len(points)):
    if labels[i] == hub_index:
        row, col = points[i]
        clustered_image[int(row), int(col)] = 255

plt.subplot(153)
plt.imshow(clustered_image, cmap='gray')
plt.title('聚类后显示轮毂')

plt.show()