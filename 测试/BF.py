import cv2
import numpy as np
import os


def orb_bf_matching():
    image_path1 = 'D:/2024study/project/car/cos/轮毂/005A.png'
    image_path2 = 'D:/2024study/project/car/cos/轮毂/005B.png'
    try:
        with open(image_path1, 'rb') as f:
            img1_data = f.read()
        img1 = cv2.imdecode(np.frombuffer(img1_data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img1 is None:
            raise Exception(f"无法读取文件 {image_path1}")

        with open(image_path2, 'rb') as f:
            img2_data = f.read()
        img2 = cv2.imdecode(np.frombuffer(img2_data, np.uint8), cv2.IMREAD_GRAYSCALE)
        if img2 is None:
            raise Exception(f"无法读取文件 {image_path2}")
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return

    # 初始化ORB检测器
    orb = cv2.ORB_create()

    # 检测关键点和计算描述符
    kp1, des1 = orb.detectAndCompute(img1, None)
    kp2, des2 = orb.detectAndCompute(img2, None)

    # 创建BF匹配器，使用NORM_L2距离
    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)

    # 进行匹配
    matches = bf.match(des1, des2)

    # 根据距离对匹配进行排序
    matches = sorted(matches, key=lambda x: x.distance)

    # 计算相似度（平均距离的倒数作为相似度指标，距离越小相似度越高）
    total_distance = sum([m.distance for m in matches])
    similarity = 1 / (total_distance / len(matches)) if matches else 0
    print(f"图像相似度: {similarity}")

    # 绘制更多匹配点（例如前50个）
    num_matches_to_draw = 50
    img3 = cv2.drawMatches(img1, kp1, img2, kp2, matches[:num_matches_to_draw], None,
                           flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    cv2.imshow('Matches', img3)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    orb_bf_matching()