import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import argrelextrema

# 解决中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 显示图像
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
matplotlib.use('TkAgg')

image = cv2.imread('img_4.png')
image = cv2.resize(image, (100, 100))
height = 100
width = 100

# 存储角度和对应的最大匹配分数
angles = []
max_vals = []

# 遍历可能的旋转角度（比如0 - 360度，步长可根据需要调整）
best_match_score = -float('inf')
best_rotated_img = None
angle = 0
best_angle = 0
for angle in range(0, 660, 1):
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_img = cv2.warpAffine(image, M, (width, height))
    # 使用模板匹配衡量旋转后图像与原图像的相似性
    result = cv2.matchTemplate(image, rotated_img, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    print(max_val, angle)
    angles.append(angle)
    max_vals.append(max_val)
    if max_val > best_match_score:
        best_angle = angle
        best_match_score = max_val
        best_rotated_img = rotated_img

print('Best match score:', best_match_score)
print('Best angle:', best_angle)

# 将角度和最大匹配分数转换为numpy数组
angles = np.array(angles)
max_vals = np.array(max_vals)

# 绘制平滑曲线
plt.plot(angles, max_vals, label='模板匹配分数')
# 使用二次样条插值平滑曲线
from scipy.interpolate import make_interp_spline
xnew = np.linspace(angles.min(), angles.max(), 300)
spl = make_interp_spline(angles, max_vals, k=2)
max_vals_smooth = spl(xnew)
plt.plot(xnew, max_vals_smooth, label='平滑后的模板匹配分数', linestyle='--')

# 寻找极大值点和极小值点
maxima_indices = argrelextrema(max_vals_smooth, np.greater)
minima_indices = argrelextrema(max_vals_smooth, np.less)

maxima_values = max_vals_smooth[maxima_indices]
minima_values = max_vals_smooth[minima_indices]

print(f"极大值点的个数: {len(maxima_indices[0])}")
print(f"极大值点的值: {maxima_values}")
print(f"极小值点的个数: {len(minima_indices[0])}")
print(f"极小值点的值: {minima_values}")

plt.xlabel('旋转角度 (度)')
plt.ylabel('模板匹配分数')
plt.title('旋转角度与模板匹配分数的关系')
plt.legend()
plt.show()