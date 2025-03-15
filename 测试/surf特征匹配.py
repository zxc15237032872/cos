import cv2
import numpy as np
import os
import matplotlib.pyplot as plt


def surf_feature_detection():
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

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 初始化SURF检测器
    surf = cv2.xfeatures2d.SURF_create()

    # 检测关键点和计算描述符
    kp, des = surf.detectAndCompute(gray, None)

    # 在图像上绘制关键点
    img_with_kp = cv2.drawKeypoints(gray, kp, img, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    plt.imshow(cv2.cvtColor(img_with_kp, cv2.COLOR_BGR2RGB))
    plt.title('SURF Feature Detection')
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    surf_feature_detection()