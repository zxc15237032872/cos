import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')


# 读取图像
image = cv2.imread('005A.png', cv2.IMREAD_GRAYSCALE)

# 检查图像是否成功读取
if image is None:
    print("无法读取图像，请检查图像路径和文件名是否正确。")
else:
    # 定义腐蚀操作的结构元素
    kernel = np.ones((4, 4), np.uint8)
    # 进行腐蚀操作
    eroded_image = cv2.erode(image, kernel, iterations=1)

    # 显示原始图像和腐蚀后的图像
    plt.figure(figsize=(10, 5))

    plt.subplot(121)
    plt.imshow(image, cmap='gray')
    plt.title('原始图像')
    plt.axis('off')

    plt.subplot(122)
    plt.imshow(eroded_image, cmap='gray')
    plt.title('腐蚀后的图像')
    plt.axis('off')

    plt.show()

    # 保存腐蚀后的图像
    cv2.imwrite('005A_eroded.png', eroded_image)
    print("腐蚀后的图像已保存为 005A_eroded.png")