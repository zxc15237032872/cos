import numpy as np
from skimage.metrics import structural_similarity as ssim
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from scipy.signal import argrelextrema
from skimage.metrics import structural_similarity as ssim

# 解决中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 显示图像
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
matplotlib.use('TkAgg')

def simplified_structural_similarity(img1, img2):
    """
    简化的结构相似度计算，只考虑结构信息
    :param img1: 二值图像 1
    :param img2: 二值图像 2
    :return: 简化的结构相似度得分
    """
    # 计算协方差
    cov = np.cov(img1.flatten(), img2.flatten())[0, 1]
    # 计算方差
    var1 = np.var(img1)
    var2 = np.var(img2)
    # 计算简化的结构相似度
    # 这里使用的公式类似于 SSIM 中结构信息的计算部分
    s = cov / (np.sqrt(var1 * var2))
    return s

# 示例二值图像
# 这里随机生成两个二值图像作为示例，实际使用时替换为你的二值图掩码

image = cv2.imread('img_4.png')
image=cv2.resize(image,(100,100))
_ , iimage = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)


# 随机生成第二个二值图像

height=100
width=100

image1= cv2.imread('img_3.png')
image1=cv2.resize(image1,(100,100))
_ ,iimage1 = cv2.threshold(image1, 127, 255, cv2.THRESH_BINARY)



# 计算简化的结构相似度
simplified_similarity = simplified_structural_similarity(iimage1, iimage)

# 计算完整的 SSIM
full_ssim = ssim(image1, image)

print(f"简化的结构相似度: {simplified_similarity}")
print(f"完整的 SSIM: {full_ssim}")

# 可视化二值图像
plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.title('Opening Mask 1')
plt.subplot(122)
plt.title('Rotated Mask 2')
plt.show()