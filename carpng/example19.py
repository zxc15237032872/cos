import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from scipy.signal import argrelextrema, savgol_filter
from scipy.misc import derivative
import time
import cv2
import numpy as np
import random

# 解决中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 显示图像
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
matplotlib.use('TkAgg')

from scipy.optimize import least_squares
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt


def ellipse_residuals(params, points):
    a, b, x0, y0, phi = params
    x = points[:, 0]
    y = points[:, 1]
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    X = (x - x0) * cos_phi + (y - y0) * sin_phi
    Y = -(x - x0) * sin_phi + (y - y0) * cos_phi
    return (X ** 2 / a ** 2 + Y ** 2 / b ** 2 - 1)


def fit_ellipse_and_transform(image):
    # 读取图像并转换为灰度图
    if image is None:
        raise ValueError("无法读取图像文件，请检查路径是否正确")
    elif not isinstance(image, np.ndarray):
        raise TypeError("输入的图像不是有效的 numpy 数组")
    original_image = image.copy()

    # 填充图像
    padding = 50
    image = cv2.copyMakeBorder(image, padding, padding, padding, padding, cv2.BORDER_CONSTANT, value=(0,))
    original_image = cv2.copyMakeBorder(original_image, padding, padding, padding, padding, cv2.BORDER_CONSTANT,
                                        value=(0,))

    # 自适应阈值处理
    thresh = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)

    # 查找轮廓
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 筛选轮廓，排除靠近边缘的轮廓
    valid_contours = []
    height, width = thresh.shape
    threshold = 10
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        if x > threshold and y > threshold and x + w < width - threshold and y + h < height - threshold:
            valid_contours.append(contour)

    # 检查 valid_contours 是否为空
    if valid_contours:
        c = max(valid_contours, key=cv2.contourArea)
        points = c.reshape(-1, 2).astype(np.float64)

        # 初始参数估计
        x0, y0 = np.mean(points, axis=0)
        a = np.max(points[:, 0]) - np.min(points[:, 0])
        b = np.max(points[:, 1]) - np.min(points[:, 1])
        phi = 0
        initial_params = [a / 2, b / 2, x0, y0, phi]

        # 使用最小二乘法拟合椭圆
        result = least_squares(ellipse_residuals, initial_params, args=(points,))
        a, b, x0, y0, phi = result.x

        # 绘制拟合的椭圆
        angle = np.rad2deg(phi)
        axes = (int(a), int(b))
        center = (int(x0), int(y0))
        cv2.ellipse(image, center, axes, angle, 0, 360, 255, 2)

        # 计算倾斜外接矩形的四个角点（使用浮点数）
        cos_phi = np.cos(phi)
        sin_phi = np.sin(phi)
        pts = np.array([
            [-a, -b],
            [a, -b],
            [a, b],
            [-a, b]
        ], dtype=np.float64)
        rotated_pts = np.zeros_like(pts)
        for i in range(4):
            rotated_pts[i, 0] = pts[i, 0] * cos_phi - pts[i, 1] * sin_phi + x0
            rotated_pts[i, 1] = pts[i, 0] * sin_phi + pts[i, 1] * cos_phi + y0

        pts2 = np.float32(rotated_pts)

        # 绘制倾斜外接矩形的四个角点
        for point in pts2:
            cv2.circle(image, (int(point[0]), int(point[1])), 5, 255, -1)

        # 连接四个角点形成倾斜矩形
        pts2_int = pts2.astype(np.int32)
        cv2.polylines(image, [pts2_int], True, 255, 2)

        # 计算能完全包含椭圆的最小正方形边长
        side_length = max(2 * a, 2 * b)

        # 目标正方形的四个角点
        pts1 = np.float32([[0, 0], [side_length, 0], [side_length, side_length], [0, side_length]])

        # 计算透视变换矩阵
        M = cv2.getPerspectiveTransform(pts2, pts1)

        try:
            # 执行透视变换，输出正方形图像
            dst = cv2.warpPerspective(original_image, M, (int(side_length), int(side_length)))

            # # 展示原始图像（含拟合椭圆和倾斜矩形）
            # plt.figure(figsize=(10, 5))
            # plt.imshow(image, cmap='gray')
            # plt.title('Original Image with Fitted Ellipse and Tilted Rectangle')
            # plt.show()

            # 在透视变换后的图像上进行处理
            _, thresh_dst = cv2.threshold(dst, 1, 255, cv2.THRESH_BINARY)
            contours_dst, _ = cv2.findContours(thresh_dst, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            if contours_dst:
                c_dst = max(contours_dst, key=cv2.contourArea)
                points_dst = c_dst.reshape(-1, 2).astype(np.float64)

                # 初始参数估计
                x0_dst, y0_dst = np.mean(points_dst, axis=0)
                a_dst = np.max(points_dst[:, 0]) - np.min(points_dst[:, 0])
                b_dst = np.max(points_dst[:, 1]) - np.min(points_dst[:, 1])
                phi_dst = 0
                initial_params_dst = [a_dst / 2, b_dst / 2, x0_dst, y0_dst, phi_dst]

                # 使用最小二乘法拟合椭圆
                result_dst = least_squares(ellipse_residuals, initial_params_dst, args=(points_dst,))
                a_dst, b_dst, x0_dst, y0_dst, phi_dst = result_dst.x

                # 创建以椭圆为形状的掩码
                mask = np.zeros(dst.shape, dtype=np.uint8)
                cv2.ellipse(mask, (int(x0_dst), int(y0_dst)), (int(a_dst), int(b_dst)), np.rad2deg(phi_dst), 0, 360,
                            255, -1)

                # 分割出目标区域（椭圆内部）
                segmented_image = cv2.bitwise_and(dst, dst, mask=mask)

                # 去除周围黑色部分
                y_indices, x_indices = np.where(segmented_image != 0)
                min_x, max_x = np.min(x_indices), np.max(x_indices)
                min_y, max_y = np.min(y_indices), np.max(y_indices)
                segmented_image = segmented_image[min_y:max_y + 1, min_x:max_x + 1]

                # # 展示透视变换后的图像（含拟合椭圆）
                # plt.figure(figsize=(10, 5))
                # plt.imshow(dst, cmap='gray')
                # plt.title('Transformed Image with Fitted Ellipse')
                # # plt.show()
                #
                # # 展示分割后的图像（仅椭圆内部且去除周围黑色）
                # plt.figure(figsize=(10, 5))
                # plt.imshow(segmented_image, cmap='gray')
                # plt.title('Segmented Image (Inside Ellipse, No Black Edges)')
                # # plt.show()
                print(segmented_image.shape)

                return segmented_image
            else:
                print("未找到有效的轮廓，请调整参数或检查图像。")
                return None
        except cv2.error as e:
            print(f"OpenCV 错误: {e}")
            return None
    else:
        print("未找到有效的轮廓，请调整筛选条件。")
        return None


def calculate_spoke_number5(mask):
    """
    计算图像的辐条数

    :param mask: 输入的图像掩码
    :return: 辐条数（离 w 最近的整数）或 None（如果未找到极大值点）
    """
    # 对二值图进行形态学操作，去除噪声和填充空洞
    # kernel = np.ones((3, 3), np.uint8)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    height, width = mask.shape[:2]

    # 存储角度和对应的模板匹配分数
    angles = []
    template_scores = []

    # 遍历可能的旋转角度（比如 0 - 360 度，步长可根据需要调整）
    for angle in range(10, 144, 1):
        M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        rotated_img = cv2.warpAffine(mask, M, (width, height))
        # 使用模板匹配衡量旋转后图像与原图像的相似性
        result = cv2.matchTemplate(mask, rotated_img, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        score = max_val
        angles.append(angle)
        template_scores.append(score)

    # 将角度和模板匹配分数转换为 numpy 数组
    angles = np.array(angles)
    template_scores = np.array(template_scores)

    # 使用 Savitzky - Golay 滤波器进行平滑处理
    window_length = 11  # 窗口长度，必须是奇数
    polyorder = 3  # 多项式阶数
    template_scores_smooth = savgol_filter(template_scores, window_length, polyorder)

    # 绘制平滑曲线
    plt.plot(angles, template_scores, label='模板匹配分数')
    plt.plot(angles, template_scores_smooth, label='平滑后的模板匹配分数', linestyle='--')
    # plt.show()

    # 寻找极大值点
    all_maxima_indices = []
    for i in range(1, len(template_scores_smooth) - 1):
        # 检查是否为局部最大值
        is_local_max = (template_scores_smooth[i] > template_scores_smooth[i - 1]) and (
                template_scores_smooth[i] > template_scores_smooth[i + 1])

        # 计算一阶导数和二阶导数
        first_derivative = derivative(lambda x: template_scores_smooth[x], i, dx=1, n=1)
        second_derivative = derivative(lambda x: template_scores_smooth[x], i, dx=1, n=2)

        # 考虑一阶导数和二阶导数都为 0 的情况
        if is_local_max or (np.isclose(first_derivative, 0) and np.isclose(second_derivative, 0)):
            all_maxima_indices.append(i)

    all_maxima_indices = np.array(all_maxima_indices)
    if len(all_maxima_indices) > 0:
        # 找到极大值中的最大值对应的索引
        max_maxima_index = all_maxima_indices[np.argmax(template_scores_smooth[all_maxima_indices])]
        # 获取对应的角度值
        corresponding_angle = angles[max_maxima_index]
        print(f"最佳匹配角度为 {corresponding_angle}")
        # 计算 w
        w = 360 / corresponding_angle
        # 找到离 w 最近的整数
        nearest_integer = round(w)
        plt.close()  # 关闭绘图窗口
        return nearest_integer
    else:
        plt.close()  # 关闭绘图窗口
        return None


def generate_mask1(corrected_image, num_clusters=2):
    """
    使用K-Means算法生成图像的掩码。

    :param corrected_image: 光照补偿后的图像
    :param num_clusters: 聚类的数量，默认为2
    :return: 生成的掩码图像
    """
    height, width = corrected_image.shape[:2]
    radius = min(height, width) // 2
    center = (width // 2, height // 2)

    circle_mask = np.zeros((height, width), dtype=np.uint8)
    cv2.circle(circle_mask, center, radius, 255, -1)

    circle_pixels = corrected_image[circle_mask == 255].reshape((-1, 1))
    circle_pixels = np.float32(circle_pixels)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    _, labels, centers = cv2.kmeans(circle_pixels, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    centers = np.uint8(centers)

    segmented_image = np.zeros((height, width), dtype=np.uint8)
    flat_result = centers[labels.flatten()].flatten()
    segmented_image[circle_mask == 255] = flat_result

    if centers[0] > centers[1]:
        mask = (segmented_image == centers[0]).astype(np.uint8) * 255
    else:
        mask = (segmented_image == centers[1]).astype(np.uint8) * 255

    return mask


def process_image_rotation_and_union(image, times, first_angle):
    """
    对输入的掩码图像进行旋转操作，并与原图进行并集操作。

    :param image: 输入的掩码图像
    :param times: 并集操作的次数
    :return: 处理后的并集图像、最佳旋转后的图像和最佳旋转角度
    """
    height, width = image.shape
    union_img = image.copy()

    for _ in range(times):
        M = cv2.getRotationMatrix2D((width / 2, height / 2), first_angle, 1)
        rotated_img = cv2.warpAffine(union_img, M, (width, height))

        # 实现并集操作
        union_img = cv2.bitwise_or(union_img, rotated_img)

    return union_img


def process_image_after_clustering(image, times, first_angle):
    """
    对聚类后的掩码图像进行处理，包括旋转和交集操作。

    :param image: 输入的掩码图像
    :param times: 交集操作的次数
    :return: 处理后的交集图像、最佳旋转后的图像和最佳旋转角度
    """
    height, width = image.shape
    intersection_img = image.copy()
    best_angle = 0
    # n=random.randint(2,10)

    for _ in range(times):
        M = cv2.getRotationMatrix2D((width / 2, height / 2), first_angle * 2, 1)
        rotated_img = cv2.warpAffine(intersection_img, M, (width, height))

        # 使用numpy的向量化操作来实现交集操作
        intersection_img = np.where((intersection_img != 0) & (rotated_img != 0),
                                    np.minimum(intersection_img, rotated_img),
                                    0)

    return intersection_img


def find_best_rotation_angle_similarity(clustering1, clustering2):
    height, width = clustering2.shape
    center = (width // 2, height // 2)
    current_max_similarity = -1
    current_best_angle = 0

    for angle in range(80, 202, 1):  # 以 1 度为步长旋转
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1)
        rotated_mask2 = cv2.warpAffine(clustering2, rotation_matrix, (width, height))

        # 使用模板匹配计算相似度
        result = cv2.matchTemplate(clustering1, rotated_mask2, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        similarity = max_val
        if similarity > current_max_similarity:
            current_max_similarity = similarity
            current_best_angle = angle

    print(f"第一次最佳旋转角度：{current_best_angle}和相似度：{current_max_similarity}")
    similarity = current_max_similarity
    for angle in np.arange(current_best_angle - 0.95, current_best_angle + 1, 0.05):  # 以 0.05 度为步长
        rotation_matrix1 = cv2.getRotationMatrix2D(center, angle, 1)
        rotated_mask2 = cv2.warpAffine(clustering2, rotation_matrix1, (width, height))

        # 使用模板匹配计算相似度
        result = cv2.matchTemplate(clustering1, rotated_mask2, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        current_similarity = max_val
        if similarity < current_similarity:
            similarity = current_similarity

    print(f"最终最佳旋转角度：{current_best_angle}和相似度：{similarity}")
    return similarity


def extract_ellipse_region(mask):
    """
    从二值图 mask 中提取并剪切出拟合椭圆部分

    :param mask: 输入的二值图
    :return: 仅包含椭圆内部且去除周围黑色的图像
    """
    # 查找轮廓
    contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        c = max(contours, key=cv2.contourArea)
        points = c.reshape(-1, 2).astype(np.float64)

        # 初始参数估计
        x0, y0 = np.mean(points, axis=0)
        a = np.max(points[:, 0]) - np.min(points[:, 0])
        b = np.max(points[:, 1]) - np.min(points[:, 1])
        phi = 0
        initial_params = [a / 2, b / 2, x0, y0, phi]

        # 使用最小二乘法拟合椭圆
        result = least_squares(ellipse_residuals, initial_params, args=(points,))
        a, b, x0, y0, phi = result.x

        # 创建以椭圆为形状的掩码
        mask_ellipse = np.zeros(mask.shape, dtype=np.uint8)
        cv2.ellipse(mask_ellipse, (int(x0), int(y0)), (int(a), int(b)), np.rad2deg(phi), 0, 360, 255, -1)

        # 分割出目标区域（椭圆内部）
        segmented_image = cv2.bitwise_and(mask, mask, mask=mask_ellipse)

        # 去除周围黑色部分
        y_indices, x_indices = np.where(segmented_image != 0)
        min_x, max_x = np.min(x_indices), np.max(x_indices)
        min_y, max_y = np.min(y_indices), np.max(y_indices)
        segmented_image = segmented_image[min_y:max_y + 1, min_x:max_x + 1]

        return segmented_image
    else:
        print("未找到有效的轮廓，请调整参数或检查图像。")
        return mask


def calculate_similarity(image_path1, image_path2):
    """
    计算两张图像的加权相似度，包括光照补偿、掩码生成、旋转处理和相似度计算。

    :param image_path1: 第一张图像的路径
    :param image_path2: 第二张图像的路径
    :return: 加权后的相似度、最佳旋转角度、第一张图像的掩码、第二张图像的掩码、第二张图像旋转后的掩码和模板匹配分数
    """
    try:
        # 读取图像
        image1 = cv2.imread(image_path1, 0)
        image2 = cv2.imread(image_path2, 0)
        g_mask1 = image1
        g_mask2 = image2
        transform1 = image1
        transform2 = image2
        resultA = 0
        resultB = 0

        if image1 is None or image2 is None:
            raise FileNotFoundError(f"无法读取图像文件: {image_path1} 或 {image_path2}")
        height, width = image1.shape
        height2, width2 = image2.shape
        # 找出高度和宽度中的最大值
        size = max(height, width, height2, width2)
        # 将 size 限制在 120 以内
        size = min(size, 120)
        u_times = 0
        c_times = 0
        # 调整图像大小

        transform1 = fit_ellipse_and_transform(image1)
        transform2 = fit_ellipse_and_transform(image2)
        transform1 = cv2.resize(transform1, (size, size), interpolation=cv2.INTER_CUBIC)
        transform2 = cv2.resize(transform2, (size, size), interpolation=cv2.INTER_CUBIC)
        # 计算光照补偿
        # 生成掩码
        g_mask1 = generate_mask1(transform1)
        g_mask2 = generate_mask1(transform2)

        resultA = calculate_spoke_number5(g_mask1.copy())
        print(resultA, "个辐条")
        resultA = resultA if resultA is not None else 0

        first_angleA = 360 / resultA if resultA != 0 else 0.0
        resultB = calculate_spoke_number5(g_mask2.copy())
        print(resultB, "个辐条")
        resultB = resultB if resultB is not None else 360

        first_angleB = 360 / resultB if resultB != 0 else 0.0
        # min = resultA if resultA > resultB else resultB

        u_times = min(resultA, resultB) // 2
        c_times = min(resultA, resultB) - 1

        print(u_times, c_times)

        # 首先使用较大的角度进行处理
        unioning1 = process_image_rotation_and_union(g_mask1, u_times, first_angleA)
        clustering1 = process_image_after_clustering(unioning1, c_times, first_angleA)

        unioning2 = process_image_rotation_and_union(g_mask2, u_times, first_angleB)
        clustering2 = process_image_after_clustering(unioning2, c_times, first_angleB)
        # clustering1 = extract_ellipse_region(clustering1)
        # clustering2 = extract_ellipse_region(clustering2)

        clustering1 = cv2.resize(clustering1, (size, size), interpolation=cv2.INTER_CUBIC)
        clustering2 = cv2.resize(clustering2, (size, size), interpolation=cv2.INTER_CUBIC)

        # clustering1=generate_mask1(clustering1)
        similarity = find_best_rotation_angle_similarity(clustering1, clustering2)
        similarity1 = similarity
        clusteringa = clustering1
        clusteringb = clustering2
        if similarity < 0.69:
            clusteringa = extract_ellipse_region(clustering1)
            clusteringb = extract_ellipse_region(clustering2)

            clusteringa = cv2.resize(clusteringa, (size, size), interpolation=cv2.INTER_CUBIC)
            clusteringb = cv2.resize(clusteringb, (size, size), interpolation=cv2.INTER_CUBIC)
            similarity1 = find_best_rotation_angle_similarity(clusteringa, clusteringb)

        if similarity1 > similarity:
            similarity = similarity1
            clustering1 = clusteringa
            clustering2 = clusteringb

        with open('xie_output2.txt', 'a', encoding='utf-8') as file:
            file.write(
                f"{image_path1},{image_path2},{resultA},{resultB},{similarity}\n")

        return clustering1, clustering2, resultA, resultB, similarity, transform1, transform2, g_mask1, g_mask2, u_times, c_times

    except Exception as e:
        print(f"计算相似度时出错: {e}")
        return None, None, None, None, None


import os
import time
output_hub_folder = 'xie6output'
image_files = [f for f in os.listdir(output_hub_folder) if f.lower().endswith('.png')]
print(len(image_files))


image_pairs = []

for i in range(len(image_files)):
    for j in range(i + 1, len(image_files)):
        # 在每个图片名前加上路径
        pair = (os.path.join(output_hub_folder, image_files[i]), os.path.join(output_hub_folder, image_files[j]))
        image_pairs.append(pair)
print(len(image_pairs))
similarity_same_type = []
similarity_different_type = []
cout = 0
for pair in image_pairs:

    image_path1, image_path2 = pair
    cout += 1
    print(f"正在计算第 {cout} 个图片对的相似度...")

    start = time.time()
    # 计算相似度、最佳旋转角度，获取掩码和旋转后的掩码以及模板匹配分数
    opening_mask1, opening_mask2, spoke_number1, spoke_number2, weighted_similarity, transform1, transform2 ,g_mask1, g_mask2,u_times, c_times= calculate_similarity(
    image_path1, image_path2)
    end = time.time()
    running_time = end - start
    # print(f"图片 {image_path1}、图片 {image_path2} 辐条数：{spoke_number1}、{spoke_number2}\t相似度：{weighted_similarity}\t运行时间：{running_time}s\n")
    with open('xie21output.txt', 'a', encoding='utf-8') as file:
        file.write(
            f"{image_path1[11:21]},{image_path2[11:21]},{spoke_number1},{spoke_number2},{weighted_similarity},{running_time}s,{u_times},{c_times}\n")


