import cv2
import numpy as np

def keep_thick_lines(image_path):
    # 读取图像
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    # 二值化处理
    _, binary_img = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY)

    # 定义结构元素
    kernel_size = 50 # 可以根据需要调整核的大小
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 进行开运算
    opened_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, kernel)

    # 从原始图像中减去开运算后的结果，得到粗线条
    thick_lines = cv2.subtract(binary_img, opened_img)

    return thick_lines

# 输入图像路径
image_path = 'img_3.png'

# 处理图像
result = keep_thick_lines(image_path)

# 显示结果
cv2.imshow('Thick Lines', result)
cv2.waitKey(0)
cv2.destroyAllWindows()

# 保存结果
cv2.imwrite('thick_lines_result.png', result)