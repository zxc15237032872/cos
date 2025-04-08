import cv2
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline
from scipy.signal import argrelextrema

# 解决中文显示问题
matplotlib.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
matplotlib.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
# 显示图像
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
matplotlib.use('TkAgg')


def calculate_spoke_number3(mask):
    """
    计算图像的辐条数

    :param mask: 输入的图像掩码
    :return: 辐条数（离 w 最近的整数）或 None（如果未找到极大值点）
    """
    # 将图像转换为灰度图，因为 ssim 通常在单通道图像上计算
    gray_image = mask
    height, width = gray_image.shape[:2]

    # 存储角度和对应的 模板匹配分数
    angles = []
    template_scores = []

    # 遍历可能的旋转角度（比如 0 - 360 度，步长可根据需要调整）
    for angle in range(3, 130, 1):
        M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        rotated_img = cv2.warpAffine(gray_image, M, (width, height))
        # 使用模板匹配衡量旋转后图像与原图像的相似性
        result = cv2.matchTemplate(gray_image, rotated_img, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        score = max_val
        angles.append(angle)
        template_scores.append(score)

    # 将角度和模板匹配分数转换为 numpy 数组
    angles = np.array(angles)
    template_scores = np.array(template_scores)

    # 绘制平滑曲线
    plt.plot(angles, template_scores, label='模板匹配分数')
    # 使用二次样条插值平滑曲线
    xnew = np.linspace(angles.min(), angles.max(), 300)
    spl = make_interp_spline(angles, template_scores, k=2)
    template_scores_smooth = spl(xnew)
    plt.plot(xnew, template_scores_smooth, label='平滑后的模板匹配分数', linestyle='--')
    # plt.show()

    # 寻找极大值点
    maxima_indices = argrelextrema(template_scores_smooth, np.greater)
    maxima_values = template_scores_smooth[maxima_indices]

    if len(maxima_values) > 0:
        # 找到极大值中的最大值对应的索引
        max_maxima_index = np.argmax(maxima_values)
        # 获取对应的角度值
        corresponding_angle = xnew[maxima_indices[0][max_maxima_index]]
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


def calculate_spoke_number(mask):
    """
    计算图像的辐条数

    :param image_path: 图像的路径
    :return: 辐条数（离 w 最近的整数）或 None（如果未找到大于 0.8 的极大值点）
    """
    # 将图像转换为灰度图，因为 ssim 通常在单通道图像上计算
    gray_image = mask
    height, width = gray_image.shape[:2]

    # 存储角度和对应的 模板匹配分数
    angles = []
    template_scores = []

    # 遍历可能的旋转角度（比如 0 - 360 度，步长可根据需要调整）
    best_match_score = -float('inf')

    for angle in range(20, 141, 1):
        M = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
        rotated_img = cv2.warpAffine(gray_image, M, (width, height))
        # 使用模板匹配衡量旋转后图像与原图像的相似性
        result = cv2.matchTemplate(gray_image, rotated_img, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
        score = max_val
        angles.append(angle)
        template_scores.append(score)
        if score > best_match_score:
            best_match_score = score

    # 将角度和模板匹配分数转换为 numpy 数组
    angles = np.array(angles)
    template_scores = np.array(template_scores)

    # 绘制平滑曲线
    plt.plot(angles, template_scores, label='模板匹配分数')
    # 使用二次样条插值平滑曲线
    xnew = np.linspace(angles.min(), angles.max(), 300)
    spl = make_interp_spline(angles, template_scores, k=2)
    template_scores_smooth = spl(xnew)
    plt.plot(xnew, template_scores_smooth, label='平滑后的模板匹配分数', linestyle='--')
    # plt.show()

    # 寻找极大值点和极小值点
    maxima_indices = argrelextrema(template_scores_smooth, np.greater)

    maxima_values = template_scores_smooth[maxima_indices]

    # 找到最大值 m
    m = np.max(maxima_values)

    valid_indices = []
    for i, value in enumerate(maxima_values):
        if m - 0.02 <= value < m:
            valid_indices.append(i)

    if valid_indices:
        # 找到对应 angle 最小的索引
        min_angle_index = valid_indices[np.argmin(xnew[maxima_indices[0][valid_indices]])]
        first_valid_maxima_index = min_angle_index
    else:
        # 找不到则取最大值 m 的索引
        first_valid_maxima_index = np.argmax(maxima_values)

    if first_valid_maxima_index is not None:
        # 获取对应的角度值
        corresponding_angle = xnew[maxima_indices[0][first_valid_maxima_index]]
        # 计算 w
        w = 360 / corresponding_angle
        # 找到离 w 最近的整数
        nearest_integer = round(w)
        plt.close()  # 关闭绘图窗口
        return nearest_integer
    else:
        plt.close()  # 关闭绘图窗口
        return None


def generate_mask(corrected_image, num_clusters=2):
    """
    使用K-Means算法生成图像的掩码。

    :param corrected_image: 光照补偿后的图像
    :param num_clusters: 聚类的数量，默认为2
    :return: 生成的掩码图像
    """
    # 将图像转换为一维数组
    pixels = corrected_image.reshape((-1, 1))
    pixels = np.float32(pixels)

    # 定义 K - Means 算法的终止条件
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)

    # 应用 K - Means 算法
    _, labels, centers = cv2.kmeans(pixels, num_clusters, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 将中心值转换为整数
    centers = np.uint8(centers)

    # 根据标签获取每个像素的聚类中心值
    segmented_image = centers[labels.flatten()]

    # 重新调整形状以匹配原始图像
    segmented_image = segmented_image.reshape(corrected_image.shape)

    # 假设轮毂对应的聚类中心值较大，获取 mask
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

    for _ in range(times):

        M = cv2.getRotationMatrix2D((width / 2, height / 2), first_angle, 1)
        rotated_img = cv2.warpAffine(intersection_img, M, (width, height))

        # 实现类似交集操作
        new_intersection_img = np.zeros_like(intersection_img)
        for y in range(height):
            for x in range(width):
                if intersection_img[y, x] != 0 and rotated_img[y, x] != 0:
                    new_intersection_img[y, x] = min(intersection_img[y, x], rotated_img[y, x])
        intersection_img = new_intersection_img
    return intersection_img


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

        if image1 is None or image2 is None:
            raise FileNotFoundError(f"无法读取图像文件: {image_path1} 或 {image_path2}")

        size = 120
        # 调整图像大小
        image1 = cv2.resize(image1, (size, size), interpolation=cv2.INTER_CUBIC)
        image2 = cv2.resize(image2, (size, size), interpolation=cv2.INTER_CUBIC)

        # 生成掩码
        mask1 = generate_mask(image1)
        mask2 = generate_mask(image2)

        resultA = calculate_spoke_number(mask1.copy())
        print(resultA)
        first_angleA = 360 / resultA
        resultB = calculate_spoke_number(mask2.copy())
        unioning1 = process_image_rotation_and_union(mask1, 2, first_angleA)
        clustering1 = process_image_after_clustering(unioning1, 5, first_angleA)

        first_angleB = 360 / resultB

        print(resultB)
        unioning2 = process_image_rotation_and_union(mask2, 2, first_angleB)
        clustering2 = process_image_after_clustering(unioning2, 5, first_angleB)

        resultA1 = calculate_spoke_number3(clustering1)
        resultB1 = calculate_spoke_number3(clustering2)

        height, width = mask2.shape
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
        with open('xie_output1', 'a', encoding='utf-8') as file:
            file.write(
                f"{image_path1[11:15]},{image_path2[11:15]},{resultA1},{resultB1},{similarity}\n")

        return clustering1, clustering2, resultA1, resultB1, similarity

    except Exception as e:
        print(f"计算相似度时出错: {e}")
        return None, None, None, None, None

#
# # 输入图片路径
# image_path1 = input("")
# image_path2 = input("")
# image_path1 = "xie_output\\" + image_path1
# image_path2 = "xie_output\\" + image_path2
#
# # 计算相似度、最佳旋转角度，获取掩码和旋转后的掩码以及模板匹配分数
# opening_mask1, opening_mask2, spoke_number1, spoke_number2, weighted_similarity = calculate_similarity(
#     image_path1, image_path2)
# print(f"图片 {image_path1}、图片 {image_path2} 辐条数：{spoke_number1}、{spoke_number2}\t相似度：{weighted_similarity}\t ")
#
# if weighted_similarity is not None:
#     # print(f"图片 {image_path1} 和 {image_path2} 的相似度为: {weighted_similarity}")
#     if weighted_similarity < 0.69:
#         print("判断结果：图片对应的轮毂已改装")
#     else:
#         print("判断结果：图片对应的轮毂未改装")
#
# # 读取原图
# img1 = cv2.imread(image_path1, 0)
# img2 = cv2.imread(image_path2, 0)
# img1 = cv2.resize(img1, (100, 100), interpolation=cv2.INTER_CUBIC)
# img2 = cv2.resize(img2, (100, 100), interpolation=cv2.INTER_CUBIC)
#
# # 显示图像和对应的 mask
# plt.figure(figsize=(8, 6))
#
# plt.subplot(2, 2, 1)
# plt.imshow(img1, cmap='gray')
# plt.title(image_path1 + '的原图')
# plt.axis('off')
#
# plt.subplot(2, 2, 2)
# plt.imshow(img2, cmap='gray')
# plt.title(image_path2 + '的原图')
# plt.axis('off')
#
# plt.subplot(2, 2, 3)
# plt.imshow(opening_mask1, cmap='gray')
# plt.title(image_path1 + '的 Mask')
# plt.axis('off')
#
# plt.subplot(2, 2, 4)
# plt.imshow(opening_mask2, cmap='gray')
# plt.title(image_path2 + '的 Mask')
# plt.axis('off')
#
import os
# plt.show()
output_hub_folder = 'xie_output'
image_files = [f for f in os.listdir(output_hub_folder) if f.lower().endswith('.png')]
image_pairs = []
for i in range(0, len(image_files), 2):
    # 在每个图片名前加上路径
    pair = (os.path.join(output_hub_folder, image_files[i]), os.path.join(output_hub_folder, image_files[i + 1]))
    # if image_files[i][0:4] == image_files[i + 1][0:4]:
        # print(f"图片 {image_files[i]} 和 {image_files[i + 1]} 属于同一类车")
    image_pairs.append(pair)
print(len(image_pairs))
similarity_same_type = []
similarity_different_type = []
cout = 0
# 遍历图片对

# output_hub_folder = 'diffoutput'
# image_files = [f for f in os.listdir(output_hub_folder) if f.lower().endswith('.png')]
# image_pairs = []
#
# for i in range(len(image_files)):
#     for j in range(i + 1, len(image_files)):
#         # 在每个图片名前加上路径
#         pair = (os.path.join(output_hub_folder, image_files[i]), os.path.join(output_hub_folder, image_files[j]))
#         image_pairs.append(pair)
# print(len(image_pairs))
# similarity_same_type = []
# similarity_different_type = []
# cout = 0
# 遍历图片对
for pair in image_pairs:
    image_path1, image_path2 = pair
    cout += 1
    # 判断是否为相同类型的车

    # # 计算相似度、最佳旋转角度，获取掩码和旋转后的掩码以及模板匹配分数
    opening_mask1, opening_mask2,spoke_number1, spoke_number2 , weighted_similarity = calculate_similarity(
        image_path1, image_path2)
    print(f"图片 {image_path1}、图片 {image_path2} 辐条数：{spoke_number1}、{spoke_number2}\t相似度：{weighted_similarity}\t ")

    if weighted_similarity is not None:
        # print(f"图片 {image_path1} 和 {image_path2} 的相似度为: {weighted_similarity}")
        if weighted_similarity < 0.69:
            print("判断结果：图片对应的轮毂已改装")
        else:
            print("判断结果：图片对应的轮毂未改装")

    # 读取原图
    img1 = cv2.imread(image_path1, 0)
    img2 = cv2.imread(image_path2, 0)
    img1 = cv2.resize(img1, (100, 100), interpolation=cv2.INTER_CUBIC)
    img2 = cv2.resize(img2, (100, 100), interpolation=cv2.INTER_CUBIC)

    # 显示图像和对应的 mask
    plt.figure(figsize=(8, 6))

    plt.subplot(2, 2, 1)
    plt.imshow(img1, cmap='gray')
    plt.title(image_path1 + '的原图')
    plt.axis('off')

    plt.subplot(2, 2, 2)
    plt.imshow(img2, cmap='gray')
    plt.title(image_path2 + '的原图')
    plt.axis('off')

    plt.subplot(2, 2, 3)
    plt.imshow(opening_mask1, cmap='gray')
    plt.title(image_path1 + '的 Mask')
    plt.axis('off')

    plt.subplot(2, 2, 4)
    plt.imshow(opening_mask2, cmap='gray')
    plt.title(image_path2 + '的 Mask')
    plt.axis('off')

    # plt.show()

    print(cout)
