import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 显示图像
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
matplotlib.use('TkAgg')

# 读取图像
# cv2.imread 函数用于读取指定路径的图像，cv2.IMREAD_GRAYSCALE 表示以灰度模式读取
image = cv2.imread('005A.png', cv2.IMREAD_GRAYSCALE)

# 检查图像是否成功读取
if image is None:
    print("无法读取图像，请检查图像路径是否正确。")
else:
    # 定义伽马值
    gamma = 1.8 # 可以调整这个值来改变变换效果，小于 1 会使图像变亮，大于 1 会使图像变暗

    # 构建伽马变换查找表
    # np.arange(256) 生成一个从 0 到 255 的数组，代表所有可能的像素值
    # 对每个像素值进行伽马变换，公式为 ((i / 255.0) ** gamma) * 255
    # 最后将结果转换为 8 位无符号整数类型
    gamma_table = np.array([((i / 255.0) ** gamma) * 255
                            for i in np.arange(256)], dtype=np.uint8)

    # 应用伽马变换
    # cv2.LUT 函数用于根据查找表对图像进行像素值替换
    gamma_transformed = cv2.LUT(image, gamma_table)

    gamma = 0.5 # 可以调整这个值来改变变换效果，小于 1 会使图像变亮，大于 1 会使图像变暗

    # 构建伽马变换查找表
    # np.arange(256) 生成一个从 0 到 255 的数组，代表所有可能的像素值
    # 对每个像素值进行伽马变换，公式为 ((i / 255.0) ** gamma) * 255
    # 最后将结果转换为 8 位无符号整数类型
    gamma_table = np.array([((i / 255.0) ** gamma) * 255
                            for i in np.arange(256)], dtype=np.uint8)

    # 应用伽马变换
    # cv2.LUT 函数用于根据查找表对图像进行像素值替换
    gamma_transformed = cv2.LUT(image, gamma_table)



    # 显示原始图像和伽马变换后的图像
    plt.subplot(121), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(gamma_transformed, cmap='gray')
    plt.title(f'Gamma Transformed (γ={gamma})'), plt.xticks([]), plt.yticks([])

    # 显示图像
    plt.show()

    # 保存伽马变换后的图像
    output_filename = f'gamma_transformed_{gamma}_005A.png'
    cv2.imwrite(output_filename, gamma_transformed)
    print(f"伽马变换后的图像已保存为 {output_filename}")