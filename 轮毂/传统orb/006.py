import cv2
import numpy as np

# 尝试使用SIFT算法，需要先安装opencv-contrib-python库
sift = cv2.SIFT_create()


def detect_circles(image):
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 中值滤波，相比高斯滤波，中值滤波在去除椒盐噪声的同时能更好地保留边缘
    blurred = cv2.medianBlur(gray, 3)

    # 自适应阈值处理，能更好地适应图像不同区域的光照变化
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    circle_centers = []
    # 遍历轮廓
    for contour in contours:
        # 计算轮廓的面积
        area = cv2.contourArea(contour)
        # 筛选合适面积范围的轮廓，可根据实际情况调整
        if 100 < area < 2000:
            # 计算轮廓的周长
            perimeter = cv2.arcLength(contour, True)
            # 计算轮廓的圆形度（圆形度接近1表示接近圆形）
            circularity = 4 * np.pi * area / (perimeter ** 2) if perimeter > 0 else 0
            # 设置圆形度的阈值，这里设为0.8，可根据实际情况调整
            if circularity > 0.8:
                # 找到最小外接圆
                (x, y), radius = cv2.minEnclosingCircle(contour)
                center = (int(x), int(y))
                circle_centers.append(center)
                radius = int(radius)

                # 绘制圆
                cv2.circle(image, center, radius, (0, 255, 0), 1)
                # 标记圆心
                cv2.circle(image, center, 2, (0, 0, 255), -1)

    return len(circle_centers), image


def detect_and_match_features(img1, img2):
    # 更精细的图像预处理：去噪和光照校正（这里简单使用高斯滤波和直方图均衡化示例）
    img1 = cv2.GaussianBlur(img1, (5, 5), 0)
    img1 = cv2.equalizeHist(cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY))
    img2 = cv2.GaussianBlur(img2, (5, 5), 0)
    img2 = cv2.equalizeHist(cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY))

    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)

    good_matches = []
    for m, n in matches:
        if m.distance < 0.6 * n.distance:  # 调整比率阈值
            good_matches.append(m)

    # 使用RANSAC算法去除错误匹配点
    if len(good_matches) >= 4:
        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        inliers = src_pts[mask.ravel() == 1]
        outlers = src_pts[mask.ravel() == 0]

        # 只保留内点作为最终的匹配点
        good_matches = [m for i, m in enumerate(good_matches) if mask.ravel()[i] == 1]
        src_pts = inliers
        dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    else:
        # 当匹配点数量不足4个时，赋空数组
        src_pts = np.array([]).reshape(-1, 1, 2)
        dst_pts = np.array([]).reshape(-1, 1, 2)

    return src_pts, dst_pts, good_matches, kp1, kp2


def rotate_image(img, angle):
    height, width = img.shape[:2]
    center = (width // 2, height // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(img, M, (width, height))

    return rotated_img


def mark_spoke(img, pts, good_matches, kp1, kp2):
    if len(pts) > 0:
        center = np.mean(pts, axis=0).flatten()
        for i, pt in enumerate(pts):
            cv2.line(img, tuple(center.astype(int)), tuple(pt.flatten().astype(int)), (0, 0, 255), 2)
            # 绘制匹配点
            cv2.drawMarker(img, tuple(kp1[good_matches[i].queryIdx].pt), color=(255, 0, 0), markerType=cv2.MARKER_CROSS)
            cv2.drawMarker(img, tuple(kp2[good_matches[i].trainIdx].pt), color=(0, 255, 0), markerType=cv2.MARKER_CROSS)
    return img


if __name__ == "__main__":
    img1 = cv2.imread('004A.png')
    img2 = cv2.imread('005A.png')
    img1 = cv2.resize(img1, (640, 640))
    img2 = cv2.resize(img2, (640, 640))

    if img1 is None or img2 is None:
        print("无法读取图像，请检查图像路径和文件名。")
    else:
        circle_count1, img1_with_circles = detect_circles(img1)
        circle_count2, img2_with_circles = detect_circles(img2)

        print(f"图1的圆心个数: {circle_count1}")
        print(f"图2的圆心个数: {circle_count2}")

        src_pts, dst_pts, good_matches, kp1, kp2 = detect_and_match_features(img1, img2)

        best_angle = 0
        best_similarity = 0
        best_rotated_img2 = None

        # 从0度到60度寻找相似度最高的角度
        for angle in range(0, 61):
            rotated_img2 = rotate_image(img2, angle)
            _, temp_dst_pts, temp_good_matches, _, _ = detect_and_match_features(img1, rotated_img2)
            similarity = len(temp_good_matches)  # 用匹配点数量作为相似度度量

            if similarity > best_similarity:
                best_similarity = similarity
                best_angle = angle
                best_rotated_img2 = rotated_img2
                dst_pts = temp_dst_pts
                good_matches = temp_good_matches

        print(f"相似度最高的角度: {best_angle} 度")
        print(f"最高相似度（匹配点数量）: {best_similarity}")

        marked_img = mark_spoke(best_rotated_img2, dst_pts, good_matches, kp1, kp2)

        # 水平合并三张图片
        combined_img = np.hstack((img1_with_circles, img2_with_circles, marked_img))

        cv2.imshow('Combined Images', combined_img)
        cv2.moveWindow('Combined Images', (cv2.getWindowProperty('Combined Images', cv2.WND_PROP_AUTOSIZE) & 0xFFF00000) // 0x1000000 - combined_img.shape[1] // 2,
                       (cv2.getWindowProperty('Combined Images', cv2.WND_PROP_AUTOSIZE) & 0xFFF000) // 0x1000 - combined_img.shape[0] // 2)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

        if circle_count1 != circle_count2 or best_similarity < 10:  # 这里的10是根据实际情况调整的匹配点数量阈值
            print("改装")
        else:
            print("没有改装")