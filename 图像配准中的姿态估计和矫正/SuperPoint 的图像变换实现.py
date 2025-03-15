import cv2
import numpy as np
import os


def get_transform_matrix(image_a, image_b):
    # 初始化SIFT检测器
    sift = cv2.SIFT_create()

    # 找到关键点和描述符
    kp1, des1 = sift.detectAndCompute(image_a, None)
    kp2, des2 = sift.detectAndCompute(image_b, None)

    # BFMatcher with default params
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k = 2)

    # 应用比值测试
    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    # 使用RANSAC算法计算变换矩阵
    M, mask = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
    return M


def transform_image(image, M, target_width, target_height):
    if M is not None:
        return cv2.warpPerspective(image, M, (target_width, target_height))
    else:
        return None


# 处理含中文路径的图像A
image_a_path = "../image/cheshen001A.png"
image_a_path = os.path.abspath(image_a_path)
image_a = cv2.imdecode(np.fromfile(image_a_path, dtype = np.uint8), cv2.IMREAD_COLOR)

# 处理含中文路径的图像B
image_b_path = "../image/cheshen001A1.jpg"
image_b_path = os.path.abspath(image_b_path)
image_b = cv2.imdecode(np.fromfile(image_b_path, dtype = np.uint8), cv2.IMREAD_COLOR)

M = get_transform_matrix(image_a, image_b)
# 手动设置目标图像大小
target_width = 800
target_height = 600
transformed_image = transform_image(image_b, M, target_width, target_height)

if transformed_image is not None:
    cv2.imshow('Transformed Image', transformed_image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
