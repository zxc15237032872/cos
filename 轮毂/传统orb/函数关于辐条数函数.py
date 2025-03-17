import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.metrics import structural_similarity as ssim
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

# 读取二值图像
image = cv2.imread('img_4.png', 0)
image = cv2.resize(image, (100, 100))
# 确保是二值图像
_, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

height, width = image.shape

# 存储模板匹配分数和 SSIM 分数
template_scores = []
ssim_scores = []

# 遍历旋转角度
for angle in range(0, 360, 1):
    M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    rotated_img = cv2.warpAffine(image, M, (width, height))

    # 模板匹配
    result = cv2.matchTemplate(image, rotated_img, cv2.TM_CCOEFF_NORMED)
    _, template_score, _, _ = cv2.minMaxLoc(result)
    template_scores.append(template_score)

    # SSIM
    ssim_score = ssim(image, rotated_img)
    ssim_scores.append(ssim_score)

# 绘制曲线
plt.plot(range(0, 360, 1), template_scores, label='模板匹配分数')
plt.plot(range(0, 360, 1), ssim_scores, label='SSIM 分数')
plt.xlabel('旋转角度 (度)')
plt.ylabel('相似度分数')
plt.title('二值图像旋转后的相似度比较')
plt.legend()
plt.show()