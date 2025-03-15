import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('005A.png', cv2.IMREAD_GRAYSCALE)
# 确保图像成功读取
if image is None:
    print("无法读取图像，请检查图像路径和文件名是否正确。")
    exit(1)

# 二值化图像
_, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# 定义击中结构元素（用于匹配目标部分）
hit_kernel = np.array([
    [0, 1, 0],
    [1, 1, 1],
    [0, 1, 0]
], dtype=np.uint8)

# 定义击不中结构元素（用于匹配背景部分）
miss_kernel = np.array([
    [1, 0, 1],
    [0, 0, 0],
    [1, 0, 1]
], dtype=np.uint8)

# 进行击中击不中变换
hit_or_miss_result = cv2.morphologyEx(binary_image, cv2.MORPH_HITMISS, hit_kernel, miss_kernel)

# 显示原始图像、二值化图像和击中击不中变换结果
plt.figure(figsize=(15, 5))

plt.subplot(131)
plt.imshow(image, cmap='gray')
plt.title('原始图像')
plt.axis('off')

plt.subplot(132)
plt.imshow(binary_image, cmap='gray')
plt.title('二值化图像')
plt.axis('off')

plt.subplot(133)
plt.imshow(hit_or_miss_result, cmap='gray')
plt.title('击中击不中变换结果')
plt.axis('off')

plt.show()

# 保存击中击不中变换结果图像
cv2.imwrite('005A_hit_or_miss_result.png', hit_or_miss_result)
print("击中击不中变换结果图像已保存为 005A_hit_or_miss_result.png")