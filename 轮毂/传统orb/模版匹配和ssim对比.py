import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import argrelextrema
from skimage.metrics import structural_similarity as ssim
from scipy.interpolate import make_interp_spline

# 解决中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 显示图像
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
matplotlib.use('TkAgg')

image = cv2.imread('005B.png')
image = cv2.resize(image, (100, 100))
# 将图像转换为灰度图，因为 ssim 通常在单通道图像上计算
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
height = 100
width = 100

# 存储角度和对应的 SSIM 分数
angles = []
ssim_scores = []

# 遍历可能的旋转角度（比如 0 - 660 度，步长可根据需要调整）
best_match_score = -float('inf')
best_rotated_img = None
angle = 0
best_angle = 0
for angle in range(0, 360, 1):
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_img = cv2.warpAffine(gray_image, M, (width, height))
    # 使用 SSIM 衡量旋转后图像与原图像的相似性
    score = ssim(gray_image, rotated_img)
    print(score, angle)
    angles.append(angle)
    ssim_scores.append(score)
    if score > best_match_score:
        best_angle = angle
        best_match_score = score
        best_rotated_img = rotated_img

print('Best match score:', best_match_score)
print('Best angle:', best_angle)

# 将角度和 SSIM 分数转换为 numpy 数组
angles = np.array(angles)
ssim_scores = np.array(ssim_scores)

# 绘制平滑曲线
plt.plot(angles, ssim_scores, label='SSIM 分数')
# 使用二次样条插值平滑曲线
xnew = np.linspace(angles.min(), angles.max(), 300)
spl = make_interp_spline(angles, ssim_scores, k=2)
ssim_scores_smooth = spl(xnew)
plt.plot(xnew, ssim_scores_smooth, label='平滑后的 SSIM 分数', linestyle='--')

# 寻找极大值点和极小值点
maxima_indices = argrelextrema(ssim_scores_smooth, np.greater)
minima_indices = argrelextrema(ssim_scores_smooth, np.less)

maxima_values = ssim_scores_smooth[maxima_indices]
minima_values = ssim_scores_smooth[minima_indices]

print(f"极大值点的个数: {len(maxima_indices[0])}")
print(f"极大值点的值: {maxima_values}")
print(f"极小值点的个数: {len(minima_indices[0])}")
print(f"极小值点的值: {minima_values}")

# 找到从左往右数第一个大于 0.6 的极大值点的索引
first_valid_maxima_index = None
for i, value in enumerate(maxima_values):
    if value > 0.3:
        first_valid_maxima_index = i
        break

if first_valid_maxima_index is not None:
    # 获取对应的角度值
    corresponding_angle = xnew[maxima_indices[0][first_valid_maxima_index]]
    # 计算 w
    w = 360 / corresponding_angle
    # 找到离 w 最近的整数
    nearest_integer = round(w)
    print(f"第一个大于 0.6 的极大值点的角度: {corresponding_angle} 度")
    print(f"360 除以该角度得到的 w 值: {w}")
    print(f"离 w 最近的整数: {nearest_integer}")
else:
    print("未找到大于 0.6 的极大值点。")

plt.xlabel('旋转角度 (度)')
plt.ylabel('SSIM 分数')
plt.title('旋转角度与 SSIM 分数的关系')
plt.legend()
plt.show()