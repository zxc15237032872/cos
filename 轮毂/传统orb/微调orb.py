import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
from PIL import Image
import cv2


def extract_orb_features(image):
    orb = cv2.ORB_create(nfeatures=1500,
                          scaleFactor=1.1,
                          nlevels=10,
                          edgeThreshold=15,
                          firstLevel=0,
                          WTA_K=3,
                          scoreType=cv2.ORB_HARRIS_SCORE,
                          patchSize=25,
                          fastThreshold=15)
    if isinstance(image, np.ndarray):
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        else:
            gray_image = image
    else:
        gray_image = np.array(image.convert('L'))
    kp, des = orb.detectAndCompute(gray_image, None)
    return kp, des


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

#
# def orb_similarity(image1, image2):
#     kp1, des1 = extract_orb_features(image1)
#     kp2, des2 = extract_orb_features(image2)
#     bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
#     matches = bf.match(des1, des2)
#     if not matches:
#         return 0
#
#     num_kp1 = len(kp1)
#     num_kp2 = len(kp2)
#     min_kp_count = min(num_kp1, num_kp2)
#
#     sorted_matches = sorted(matches, key=lambda x: x.distance)
#     weighted_sum = 0
#     total_weight = 0
#
#     for i, match in enumerate(sorted_matches):
#         weight = 1 / (i + 1)  # 距离越近权重越高
#         weighted_sum += weight * (256 - match.distance)
#         total_weight += weight
#
#     similarity_score = weighted_sum / total_weight if total_weight > 0 else 0
#     similarity_percentage = (similarity_score / 256) * min_kp_count / min_kp_count * 100
#     return similarity_percentage


def orb_similarity(image1, image2):
    kp1, des1 = extract_orb_features(image1)
    kp2, des2 = extract_orb_features(image2)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)
    if not matches:
        return 0

    num_kp1 = len(kp1)
    num_kp2 = len(kp2)
    min_kp_count = min(num_kp1, num_kp2)

    total_distance = sum([match.distance for match in matches])
    avg_distance = total_distance / len(matches) if matches else 0

    # 这里我们假设最大可能平均距离为256（当所有匹配点距离都最大时）
    similarity_based_on_distance = 1 - avg_distance / 256
    similarity_based_on_count = len(matches) / min_kp_count

    # 综合距离和数量的相似度
    combined_similarity = (similarity_based_on_distance + similarity_based_on_count) / 2
    similarity_percentage = combined_similarity * 100

    return similarity_percentage

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

    kp1, des1 = extract_orb_features(enhanced_preprocess(image1))
    kp2, des2 = extract_orb_features(enhanced_preprocess(image2))

    similarity = orb_similarity(enhanced_preprocess(image1), enhanced_preprocess(image2))

    print(f"图像对 ({image_path1}, {image_path2}) 的相似度: {similarity:.2f}%")
    print(f"图像 {image_path1} 的关键点个数: {len(kp1)}")
    print(f"图像 {image_path2} 的关键点个数: {len(kp2)}")
