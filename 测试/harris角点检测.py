import cv2
import numpy as np
import matplotlib.pyplot as plt


def harris_corner_detection():
    # 读取图像
    img = cv2.imread('')
    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    # Harris角点检测
    dst = cv2.cornerHarris(gray, 2, 3, 0.04)

    # 结果进行膨胀，便于标记角点
    dst = cv2.dilate(dst, None)

    # 设定阈值，标记角点
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Harris Corner Detection')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    harris_corner_detection()