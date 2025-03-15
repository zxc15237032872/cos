import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# 尝试使用SIFT算法，需要先安装opencv-contrib-python库
sift = cv2.SIFT_create()


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

    return src_pts, dst_pts


def rotate_image(img, angle):
    height, width = img.shape[:2]
    center = (width // 2, height // 2)

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img = cv2.warpAffine(img, M, (width, height))

    return rotated_img


import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.feature import local_binary_pattern
from sklearn.preprocessing import normalize

def compare_similarity(img1, img2):
    # 图像预处理：自适应直方图均衡化和边缘增强
    def preprocess_image(img):
        # 转换为灰度图
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 自适应直方图均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        # 边缘增强（使用拉普拉斯算子）
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian = cv2.convertScaleAbs(laplacian)
        return laplacian

    img1_processed = preprocess_image(img1)
    img2_processed = preprocess_image(img2)

    # 特征提取：结合SIFT和LBP特征
    def extract_features(img):
        # SIFT特征提取
        sift = cv2.SIFT_create()
        kp, des = sift.detectAndCompute(img, None)
        if des is None:
            des = np.array([])
        # LBP特征提取
        lbp = local_binary_pattern(img, 8, 1, method='uniform')
        hist, _ = np.histogram(lbp.ravel(), bins=np.arange(0, 10), range=(0, 10))
        hist = normalize(hist.reshape(1, -1), norm='l2').flatten()
        return des, hist

    des1, lbp_hist1 = extract_features(img1_processed)
    des2, lbp_hist2 = extract_features(img2_processed)

    # 特征匹配：使用SIFT特征进行匹配
    if len(des1) > 0 and len(des2) > 0:
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des1, des2, k=2)
        good_matches = []
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good_matches.append(m)
        sift_similarity = len(good_matches) / max(len(des1), len(des2))
    else:
        sift_similarity = 0

    # 计算LBP特征的相似度
    lbp_similarity = 1 - np.linalg.norm(lbp_hist1 - lbp_hist2)

    # 计算SSIM相似度
    ssim_similarity = ssim(img1_processed, img2_processed)

    # 综合相似度：可以根据实际情况调整权重
    combined_similarity = 0.4 * sift_similarity + 0.3 * lbp_similarity + 0.3 * ssim_similarity

    return combined_similarity


def mark_spoke(img, pts):
    if len(pts) > 0:
        center = np.mean(pts, axis=0).flatten()
        for pt in pts:
            cv2.line(img, tuple(center.astype(int)), tuple(pt.flatten().astype(int)), (0, 0, 255), 2)
    return img


if __name__ == "__main__":
    img1 = cv2.imread('005A.png')
    img2 = cv2.imread('005B.png')

    # 调整图像尺寸为相同大小
    img1=cv2.resize(img1,(50,50))
    img2 = cv2.resize(img2, (50, 50))

    src_pts, dst_pts = detect_and_match_features(img1, img2)

    best_angle = 35  # 初始假设角度
    best_similarity = 0
    best_rotated_img2 = None

    # 在围绕35度的一定范围内寻找相似度最高的角度
    angle_range = 10  # 角度范围，比如在35度上下10度内搜索
    step = 1  # 角度步长

    for angle in range(0, 70, step):
        rotated_img2 = rotate_image(img2, angle)
        similarity = compare_similarity(img1, rotated_img2)

        if similarity > best_similarity:
            best_similarity = similarity
            best_angle = angle
            best_rotated_img2 = rotated_img2

    print(f"相似度最高的角度: {best_angle} 度")
    print(f"最高相似度: {best_similarity}")

    if best_rotated_img2 is not None:
        marked_img = mark_spoke(best_rotated_img2, dst_pts)

        cv2.imshow('Original Image 1', img1)
        cv2.imshow('Best Rotated Image 2', marked_img)

        # 获取屏幕尺寸
        screen_size = (cv2.getWindowImageRect('Original Image 1')[2], cv2.getWindowImageRect('Original Image 1')[3])
        img1_height, img1_width = img1.shape[:2]
        img2_height, img2_width = marked_img.shape[:2]

        # 计算窗口位置
        x1 = (screen_size[0] - img1_width) // 2
        y1 = (screen_size[1] - img1_height) // 2
        x2 = (screen_size[0] - img2_width) // 2
        y2 = (screen_size[1] - img2_height) // 2

        # 设置窗口位置
        cv2.moveWindow('Original Image 1', x1, y1)
        cv2.moveWindow('Best Rotated Image 2', 500,300)

        cv2.waitKey(0)
        cv2.destroyAllWindows()