import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
def tomasi_corner_detection():
    # 图像路径
    image_path = 'D:/2024study/project/car/cos/image/网格图.jpg'
    try:
        with open(image_path, 'rb') as f:
            img_data = f.read()
        img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            raise Exception(f"无法读取文件 {image_path}")
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return

    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 使用Tomasi角点检测，maxCorners表示最多检测到的角点数，qualityLevel为角点质量水平，minDistance是角点之间的最小距离
    corners = cv2.goodFeaturesToTrack(gray, maxCorners=1000, qualityLevel=0.01, minDistance=10)
    corners = np.int0(corners)
    for i in corners:
        x, y = i.ravel()
        cv2.circle(img, (x, y), 3, 255, -1)
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title('Tomasi Corner Detection')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    tomasi_corner_detection()