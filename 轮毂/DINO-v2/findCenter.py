import cv2
import numpy as np
import random
import math
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


class RANSAC:
    def __init__(self, data, threshold, P, S, N):
        self.point_data = data  # 椭圆轮廓点集
        self.length = len(self.point_data)  # 椭圆轮廓点集长度
        self.error_threshold = threshold  # 模型评估误差容忍阀值

        self.N = N  # 随机采样数
        self.S = S  # 设定的内点比例
        self.P = P  # 采得 N 点去计算的正确模型概率
        self.max_inliers = self.length * self.S  # 设定最大内点阀值
        self.items = 10

        self.count = 0  # 内点计数器
        self.best_model = ((0, 0), (1e-6, 1e-6), 0)  # 椭圆模型存储器

    def random_sampling(self, n):
        """随机取 n 个数据点"""
        return np.array(random.sample(list(self.point_data), n))

    def Geometric2Conic(self, ellipse):
        """计算椭圆方程系数"""
        # Ax ^ 2 + Bxy + Cy ^ 2 + Dx + Ey + F
        (x0, y0), (bb, aa), phi_b_deg = ellipse

        a, b = aa / 2, bb / 2  # Semimajor and semiminor axes
        phi_b_rad = phi_b_deg * np.pi / 180.0  # Convert phi_b from deg to rad
        ax, ay = -np.sin(phi_b_rad), np.cos(phi_b_rad)  # Major axis unit vector

        # Useful intermediates
        a2 = a * a
        b2 = b * b

        # Conic parameters
        if a2 > 0 and b2 > 0:
            A = ax * ax / a2 + ay * ay / b2
            B = 2 * ax * ay / a2 - 2 * ax * ay / b2
            C = ay * ay / a2 + ax * ax / b2
            D = (-2 * ax * ay * y0 - 2 * ax * ax * x0) / a2 + (2 * ax * ay * y0 - 2 * ay * ay * x0) / b2
            E = (-2 * ax * ay * x0 - 2 * ay * ay * y0) / a2 + (2 * ax * ay * x0 - 2 * ax * ax * y0) / b2
            F = (2 * ax * ay * x0 * y0 + ax * ax * x0 * x0 + ay * ay * y0 * y0) / a2 + \
                (-2 * ax * ay * x0 * y0 + ay * ay * x0 * x0 + ax * ax * y0 * y0) / b2 - 1
        else:
            # Tiny dummy circle - response to a2 or b2 == 0 overflow warnings
            A, B, C, D, E, F = (1, 0, 1, 0, 0, -1e-6)

        # Compose conic parameter array
        conic = np.array((A, B, C, D, E, F))
        return conic

    def eval_model(self, ellipse):
        """评估椭圆模型，统计内点个数"""
        # this an ellipse ?
        a, b, c, d, e, f = self.Geometric2Conic(ellipse)
        E = 4 * a * c - b * b
        if E <= 0:
            return 0, 0

        #  which long axis ?
        (x, y), (LAxis, SAxis), Angle = ellipse
        LAxis, SAxis = LAxis / 2, SAxis / 2
        if SAxis > LAxis:
            temp = SAxis
            SAxis = LAxis
            LAxis = temp

        # calculate focus
        Axis = math.sqrt(LAxis * LAxis - SAxis * SAxis)
        f1_x = x - Axis * math.cos(Angle * math.pi / 180)
        f1_y = y - Axis * math.sin(Angle * math.pi / 180)
        f2_x = x + Axis * math.cos(Angle * math.pi / 180)
        f2_y = y + Axis * math.sin(Angle * math.pi / 180)

        # identify inliers points
        f1, f2 = np.array([f1_x, f1_y]), np.array([f2_x, f2_y])
        f1_distance = np.square(self.point_data - f1)
        f2_distance = np.square(self.point_data - f2)
        all_distance = np.sqrt(f1_distance[:, 0] + f1_distance[:, 1]) + np.sqrt(f2_distance[:, 0] + f2_distance[:, 1])

        Z = np.abs(2 * LAxis - all_distance)
        delta = math.sqrt(np.sum((Z - np.mean(Z)) ** 2) / len(Z))

        # Update inliers set
        inliers = np.nonzero(Z < 0.8 * delta)[0]
        inlier_pnts = self.point_data[inliers]

        return len(inlier_pnts), inlier_pnts

    def execute_ransac(self):
        max_iterations = 1000  # 设置最大迭代次数
        iteration = 0
        while iteration < max_iterations:
            # 1.select N points at random
            select_points = self.random_sampling(self.N)

            # 2.fitting N ellipse points
            ellipse = cv2.fitEllipse(select_points)

            # 3.assess model and calculate inliers points
            inliers_count, inliers_set = self.eval_model(ellipse)

            # 4.number of new inliers points more than number of old inliers points ?
            if inliers_count > self.count and len(inliers_set) >= 5:
                ellipse_ = cv2.fitEllipse(inliers_set)  # fitting ellipse for inliers points
                self.count = inliers_count  # Update inliers set
                self.best_model = ellipse_  # Update best ellipse

                # 5.number of inliers points reach the expected value
                if self.count > self.max_inliers:
                    print('the number of inliers: ', self.count)
                    break

            iteration += 1

        return self.best_model

if __name__ == '__main__':
    # 读取图像
    img = cv2.imread('004A.png')
    if img is None:
        print("无法读取图像，请检查图像路径和文件名是否正确。")
        exit()

    # 转换为灰度图像
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 高斯模糊以减少噪声
    blurred = cv2.GaussianBlur(gray, (9, 9), 2)
    # 边缘检测
    edges = cv2.Canny(blurred, 50, 150)

    # 查找轮廓
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        if len(contour) > 5:  # 拟合椭圆至少需要 5 个点
            points_data = np.reshape(contour, (-1, 2))
            # Ransac 拟合
            Ransac = RANSAC(data=points_data, threshold=0.5, P=0.99, S=0.618, N=10)
            (X, Y), (LAxis, SAxis), Angle = Ransac.execute_ransac()

            # 绘制拟合的椭圆
            cv2.ellipse(img, ((X, Y), (LAxis, SAxis), Angle), (0, 255, 0), 2)
            # 绘制圆心
            cv2.circle(img, (int(X), int(Y)), 5, (0, 0, 255), -1)

    # 显示结果图像
    cv2.imshow('Result', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()