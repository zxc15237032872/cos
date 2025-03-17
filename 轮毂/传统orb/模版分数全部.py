import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# 解决中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 显示图像
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
matplotlib.use('TkAgg')
image = cv2.imread('img_4.png')
image=cv2.resize(image,(100,100))
height=100
width=100

# 遍历可能的旋转角度（比如0 - 360度，步长可根据需要调整）
best_match_score = -float('inf')
best_rotated_img = None
angle = 0
best_angle = 0
for angle in range(0, 300, 1):
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_img = cv2.warpAffine(image, M, (width, height))
    # 使用模板匹配衡量旋转后图像与原图像的相似性
    result = cv2.matchTemplate(image, rotated_img, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    print( max_val, angle)
    if max_val > best_match_score:
        best_angle = angle
        best_match_score = max_val
        best_rotated_img = rotated_img

print('Best match score:', best_match_score)
print('Best angle:', best_angle)