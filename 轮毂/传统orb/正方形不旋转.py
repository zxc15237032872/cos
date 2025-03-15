import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
from PIL import Image
import cv2


def convert_to_square(image, size = 90):
    gray_image = image.convert('L')
    img_array = np.array(gray_image)
    rows, cols = np.nonzero(img_array)
    if len(rows) == 0 or len(cols) == 0:
        return Image.new("RGB", (size, size), (0, 0, 0))
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    major_axis = max(max_row - min_row, max_col - min_col)
    minor_axis = min(max_row - min_row, max_col - min_col)
    scale_ratio = major_axis / minor_axis if minor_axis!= 0 else 1
    if max_row - min_row > max_col - min_col:
        new_width = int(image.width * scale_ratio)
        resized_image = image.resize((new_width, image.height), resample = Image.BICUBIC)
    else:
        new_height = int(image.height * scale_ratio)
        resized_image = image.resize((image.width, new_height), resample = Image.BICUBIC)
    gray_resized = resized_image.convert('L')
    resized_array = np.array(gray_resized)
    rows, cols = np.nonzero(resized_array)
    min_row, max_row = np.min(rows), np.max(rows)
    min_col, max_col = np.min(cols), np.max(cols)
    left = min_col
    top = min_row
    right = max_col
    bottom = max_row
    cropped_image = resized_image.crop((left, top, right, bottom))
    final_image = cropped_image.resize((size, size), resample = Image.BICUBIC)
    return final_image


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
    orb = cv2.ORB_create(nfeatures = 150,  # 增加期望检测的最大关键点数量
                          scaleFactor = 1.6,  # 更精细的尺度因子
                          nlevels = 1,  # 增加图像金字塔层数
                          edgeThreshold = 14,  # 降低边缘阈值
                          firstLevel = 0,
                          WTA_K = 3,  # 改变生成描述符的点数
                          scoreType = cv2.ORB_HARRIS_SCORE,
                          patchSize = 22,  # 减小patchSize
                          fastThreshold = 15)
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
    square_image1 = convert_to_square(image1)
    square_image2 = convert_to_square(image2)

    kp1, des1 = extract_orb_features(square_image1)
    kp2, des2 = extract_orb_features(square_image2)

    similarity = orb_similarity(square_image1, square_image2)
    max_possible_matches = min(len(kp1), len(kp2))
    similarity_percentage = (similarity / max_possible_matches) * 100 if max_possible_matches > 0 else 0
    print(f"图像 {image_path1} 和 {image_path2} 的相似度: {similarity_percentage:.2f}%")

    # print(f"图像 {image_path1} 的关键点个数: {len(kp1)}")
    # print(f"图像 {image_path2} 的关键点个数: {len(kp2)}")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x: x.distance)

    img1_with_kp = cv2.drawKeypoints(np.array(square_image1), kp1, None, color = (0, 255, 0), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_with_kp = cv2.drawKeypoints(np.array(square_image2), kp2, None, color = (0, 255, 0), flags = cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2[img2_idx].pt
        cv2.line(img1_with_kp, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)

    combined_image = np.hstack((img1_with_kp, img2_with_kp))

    # plt.figure(figsize = (12, 6))
    # plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
    # plt.title(f"图像 {image_path1} 和 {image_path2} 的关键点匹配")
    # plt.show()
