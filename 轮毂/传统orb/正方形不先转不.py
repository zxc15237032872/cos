import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
from PIL import Image
import cv2


def sharpen_image(image):
    laplacian = cv2.Laplacian(image, cv2.CV_64F)
    laplacian = np.uint8(np.absolute(laplacian))
    sharpened = float(1.5) * image - float(0.5) * laplacian
    sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
    sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
    sharpened = sharpened.round().astype(np.uint8)
    return sharpened


def enhanced_preprocess(image):
    # 直方图均衡化增强对比度
    img = np.array(image.convert('L'))
    equalized = cv2.equalizeHist(img)
    # 再次锐化
    sharpened = sharpen_image(equalized)
    return sharpened


def extract_orb_features(image):
    # orb = cv2.ORB_create(nfeatures = 200,
    #                       scaleFactor = 1.6,
    #                       nlevels = 1,
    #                       edgeThreshold = 14,
    #                       firstLevel = 0,
    #                       WTA_K = 3,
    #                       scoreType = cv2.ORB_HARRIS_SCORE,
    #                       patchSize = 10,
    #                       fastThreshold = 15)
    # 创建ORB特征检测器和描述符提取器对象
    # nfeatures：指定ORB算法期望检测到的最大关键点数量，这里设置为200
    # 意味着最多会检测并返回200个关键点
    orb = cv2.ORB_create(nfeatures=200,
                         # scaleFactor：指定图像金字塔中每层图像之间的尺度因子
                         # 这里设置为1.6，表示每层图像的尺度是上一层的1.6倍
                         scaleFactor=1.2,
                         # nlevels：指定图像金字塔的层数，这里设置为1
                         # 表示只使用原始图像，不构建多层图像金字塔
                         nlevels=1,
                         # edgeThreshold：指定边缘阈值
                         # 距离图像边缘小于此阈值的区域内不会检测关键点，这里设置为14
                         edgeThreshold=14,
                         # firstLevel：指定图像金字塔中第一层的索引，通常设置为0
                         firstLevel=0,
                         # WTA_K：指定生成一个ORB描述符时，比较的点数
                         # 这里设置为3，表示使用3个点来生成一个描述符
                         WTA_K=3,
                         # scoreType：指定用于对关键点进行排序的分数类型
                         # 这里使用cv2.ORB_HARRIS_SCORE，表示使用Harris角点响应函数来计算关键点分数
                         scoreType=cv2.ORB_HARRIS_SCORE,
                         # patchSize：指定用于计算ORB描述符的邻域大小
                         # 这里设置为10，表示以关键点为中心，10x10大小的邻域用于计算描述符
                         patchSize=9,
                         # fastThreshold：指定FAST角点检测的阈值
                         # 这里设置为15，表示当像素与周围16个像素中的12个像素的灰度值差异大于此阈值时，该像素被认为是角点
                         fastThreshold=11)
    img = enhanced_preprocess(image)
    kp, des = orb.detectAndCompute(img, None)
    return kp, des


def orb_similarity(image1, image2):
    kp1, des1 = extract_orb_features(image1)
    kp2, des2 = extract_orb_features(image2)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = bf.match(des1, des2)
    similarity = len(matches)
    return similarity


image_pairs = [
    ('006A.png', '006A.png'),
    ('006A.png', '006A1.png'),
    ('004A.png', '004A1.png'),
    ('007A.png', '007A1.png'),
    ('005A.png', '005B.png'),
    ('006A.png', '004A1.png'),
    ('004A.png', '005A.png'),
    ('005A.png', '006A.png')
]

for pair in image_pairs:
    image_path1, image_path2 = pair
    image1 = Image.open(image_path1).convert('RGB')
    image2 = Image.open(image_path2).convert('RGB')

    kp1, des1 = extract_orb_features(image1)
    kp2, des2 = extract_orb_features(image2)

    # 输出关键点个数
    print(f"图像 {image_path1} 的关键点个数: {len(kp1)}")
    print(f"图像 {image_path2} 的关键点个数: {len(kp2)}")

    similarity = orb_similarity(image1, image2)
    max_possible_matches = min(len(kp1), len(kp2))
    similarity_percentage = (similarity / max_possible_matches) * 100 if max_possible_matches > 0 else 0
    print(f"图像 {image_path1} 和 {image_path2} 的相似度: {similarity_percentage:.2f}%")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x: x.distance)

    img1_with_kp = cv2.drawKeypoints(np.array(image1), kp1, None, color = (0, 255, 0), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_with_kp = cv2.drawKeypoints(np.array(image2), kp2, None, color = (0, 255, 0), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    # 获取较大图像的高度
    # max_height = max(img1_with_kp.shape[0], img2_with_kp.shape[0])
    # # 调整图像大小
    # img1_with_kp = cv2.resize(img1_with_kp, (int(img1_with_kp.shape[1] * max_height / img1_with_kp.shape[0]), max_height))
    # img2_with_kp = cv2.resize(img2_with_kp, (int(img2_with_kp.shape[1] * max_height / img2_with_kp.shape[0]), max_height))
    # 获取较大图像的高度
    max_height = max(img1_with_kp.shape[0], img2_with_kp.shape[0])

    if img1_with_kp.shape[0] < max_height:
        height_diff = max_height - img1_with_kp.shape[0]
        top_padding = height_diff // 2
        bottom_padding = height_diff - top_padding
        img1_with_kp = cv2.copyMakeBorder(img1_with_kp, top_padding, bottom_padding, 0, 0, cv2.BORDER_CONSTANT,
                                          value=(0, 0, 0))

    if img2_with_kp.shape[0] < max_height:
        height_diff = max_height - img2_with_kp.shape[0]
        top_padding = height_diff // 2
        bottom_padding = height_diff - top_padding
        img2_with_kp = cv2.copyMakeBorder(img2_with_kp, top_padding, bottom_padding, 0, 0, cv2.BORDER_CONSTANT,
                                          value=(0, 0, 0))


    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt
        # 根据调整后的图像尺寸，调整坐标
        x1 = x1 * max_height / img1_with_kp.shape[0]
        y1 = y1 * max_height / img1_with_kp.shape[0]
        x2 = x2 * max_height / img2_with_kp.shape[0] + img1_with_kp.shape[1]
        y2 = y2 * max_height / img2_with_kp.shape[0]
        cv2.line(img1_with_kp, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)

    combined_image = np.hstack((img1_with_kp, img2_with_kp))

    # plt.figure(figsize = (12, 6))
    # plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
    # plt.title(f"图像 {image_path1} 和 {image_path2} 的关键点匹配")
    # plt.show()
