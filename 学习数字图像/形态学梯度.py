import cv2
import numpy as np
import matplotlib.pyplot as plt
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

# 二值化图像（也可以根据需要不进行二值化，直接对灰度图操作）
# _, binary_image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

# 定义结构元素
kernel = np.ones((2, 2), np.uint8)

# 计算形态学梯度（基本梯度：膨胀 - 腐蚀）
dilated = cv2.dilate(image, kernel, iterations=1)
eroded = cv2.erode(image, kernel, iterations=1)
gradient_image = cv2.subtract(dilated, eroded)

# 显示原始图像和形态学梯度图像
plt.figure(figsize=(10, 5))

plt.subplot(121)
plt.imshow(image, cmap='gray')
plt.title('原始图像')
plt.axis('off')

plt.subplot(122)
plt.imshow(gradient_image, cmap='gray')
plt.title('形态学梯度图像')
plt.axis('off')

plt.show()

# 保存形态学梯度图像
cv2.imwrite('005A_gradient_result.png', gradient_image)
print("形态学梯度图像已保存为 005A_gradient_result.png")