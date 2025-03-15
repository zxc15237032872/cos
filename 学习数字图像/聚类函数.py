def generate_mask(corrected_image, num_clusters=2):
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