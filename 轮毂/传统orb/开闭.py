# import cv2
# import numpy as np
# import matplotlib.pyplot as plt
# # 定义模板匹配函数
# import matplotlib
# matplotlib.use('TkAgg')
#
#
# # 读取图片
# img = cv2.imread('kb.png', 0)
# height, width = img.shape
#
# # 遍历可能的旋转角度（比如0 - 360度，步长可根据需要调整）
# best_match_score = -float('inf')
# best_rotated_img = None
# angle = 0
# best_angle = 0
# for angle in range(10, 300, 1):
#     M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
#     rotated_img = cv2.warpAffine(img, M, (width, height))
#     # 使用模板匹配衡量旋转后图像与原图像的相似性
#     result = cv2.matchTemplate(img, rotated_img, cv2.TM_CCOEFF_NORMED)
#     min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
#     if max_val > best_match_score:
#         best_angle = angle
#         best_match_score = max_val
#         best_rotated_img = rotated_img
# print('Best match score:', best_match_score)
# print('Best angle:', best_angle)
# plt.figure(figsize=(10, 10))
# plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Original Image')
# plt.subplot(132), plt.imshow(best_rotated_img, cmap='gray'), plt.title('Rotated Image')
#
#
#
# # 将最佳旋转图像与原图像求平均
# averaged_img = (img + best_rotated_img) / 2
# kernel = np.ones((3, 3), np.uint8)
# opening = cv2.morphologyEx(averaged_img.astype(np.uint8), cv2.MORPH_OPEN, kernel)
# closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)
# plt.subplot(133), plt.imshow(closing, cmap='gray'), plt.title('closed Image')
#
# plt.show()
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# 读取图片
img = cv2.imread('kb.png', 0)
height, width = img.shape

# 遍历可能的旋转角度（比如0 - 360度，步长可根据需要调整）
best_match_score = -float('inf')
best_rotated_img = None
angle = 0
best_angle = 0
for angle in range(10, 300, 1):
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_img = cv2.warpAffine(img, M, (width, height))
    # 使用模板匹配衡量旋转后图像与原图像的相似性
    result = cv2.matchTemplate(img, rotated_img, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    if max_val > best_match_score:
        best_angle = angle
        best_match_score = max_val
        best_rotated_img = rotated_img

print('Best match score:', best_match_score)
print('Best angle:', best_angle)

# 实现类似交集操作
intersection_img = np.zeros_like(img)
for y in range(height):
    for x in range(width):
        if img[y, x] != 0 and best_rotated_img[y, x] != 0:
            intersection_img[y, x] = min(img[y, x], best_rotated_img[y, x])

plt.figure(figsize=(10, 10))
plt.subplot(131), plt.imshow(img, cmap='gray'), plt.title('Original Image')
plt.subplot(132), plt.imshow(best_rotated_img, cmap='gray'), plt.title('Rotated Image')
plt.subplot(133), plt.imshow(intersection_img, cmap='gray'), plt.title('Intersection Image')

plt.show()