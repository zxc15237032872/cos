import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')

# 读取图像
# cv2.imread() 函数用于读取指定路径的图像文件
# 第二个参数 cv2.IMREAD_GRAYSCALE 表示以灰度模式读取图像，因为对数变换通常在单通道（灰度）图像上进行效果更好
image = cv2.imread('005A.png', cv2.IMREAD_GRAYSCALE)

# 检查图像是否成功读取
# 如果图像读取失败，cv2.imread() 会返回 None
if image is None:
    print("无法读取图像，请检查图像路径是否正确。")
else:
    # 对数变换公式：s = c * log(1 + r)
    # 其中 s 是变换后的像素值，r 是原始像素值，c 是一个常数，用于调整变换的幅度
    # 这里我们选择 c = 255 / log(1 + max(r))，max(r) 是图像中像素的最大值（通常为 255）
    c = 255 / np.log(1 + np.max(image))

    # 进行对数变换
    # np.log() 函数用于计算每个像素值加 1 后的自然对数
    # 乘以常数 c 后将结果转换为 uint8 类型，因为图像像素值通常是 8 位无符号整数
    log_transformed = c * np.log(1 + image)
    log_transformed = np.array(log_transformed, dtype=np.uint8)

    # 显示原始图像和对数变换后的图像
    # 使用 matplotlib 库的 subplot 函数创建一个 1 行 2 列的图像布局
    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(log_transformed, cmap='gray')
    plt.title('Log Transformed Image'), plt.xticks([]), plt.yticks([])

    # 显示图像
    plt.show()

    # 保存对数变换后的图像
    # cv2.imwrite() 函数用于将图像保存到指定路径
    cv2.imwrite('log_transformed_005A.png', log_transformed)
    print("对数变换后的图像已保存为 log_transformed_005A.png")