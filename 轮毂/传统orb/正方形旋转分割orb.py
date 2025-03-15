import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import cv2


def convert_to_square(image, size=90):
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
        resized_image = image.resize((new_width, image.height), resample=Image.BICUBIC)
    else:
        new_height = int(image.height * scale_ratio)
        resized_image = image.resize((image.width, new_height), resample=Image.BICUBIC)
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
    final_image = cropped_image.resize((size, size), resample=Image.BICUBIC)
    center = (size // 2, size // 2)
    radius = size // 2
    final_array = np.array(final_image)
    for y in range(size):
        for x in range(size):
            if (x - center[0]) ** 2 + (y - center[1]) ** 2 > radius ** 2:
                final_array[y, x] = [0, 0, 0]
    final_image = Image.fromarray(final_array)
    return final_image


def rotate_image_opencv(image, angle):
    img_array = np.array(image)
    height, width = img_array.shape[:2]
    center = (width // 2, height // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated_img_array = cv2.warpAffine(img_array, rotation_matrix, (width, height), borderMode=cv2.BORDER_REPLICATE)
    size = min(width, height)
    center = (width // 2, height // 2)
    radius = size // 2
    for y in range(height):
        for x in range(width):
            if (x - center[0]) ** 2 + (y - center[1]) ** 2 > radius ** 2:
                rotated_img_array[y, x] = [0, 0, 0]
    rotated_image = Image.fromarray(rotated_img_array)
    return rotated_image


def calculate_ssim(image1, image2):
    img1_array = np.array(image1.convert('L'))
    img2_array = np.array(image2.convert('L'))
    return ssim(img1_array, img2_array)


def find_best_rotation_angle(image_path1, image_path2):
    image1 = Image.open(image_path1).convert('RGB')
    image2 = Image.open(image_path2).convert('RGB')
    square_image1 = convert_to_square(image1)
    square_image2 = convert_to_square(image2)
    max_similarity = -1
    best_angle = 0
    best_rotated_image2 = None
    for angle in np.arange(0, 100, 1):
        rotated_image2 = rotate_image_opencv(square_image2, angle)
        similarity = calculate_ssim(square_image1, rotated_image2)
        if similarity > max_similarity:
            max_similarity = similarity
            best_angle = angle
            best_rotated_image2 = rotated_image2
    print(f"图片 {image_path1} 和 {image_path2} 的最佳旋转角度为: {best_angle} 度")
    return square_image1, square_image2, best_rotated_image2, best_angle


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
    orb = cv2.ORB_create(nfeatures=1500,  # 增加期望检测的最大关键点数量
                         scaleFactor=1.1,  # 更精细的尺度因子
                         nlevels=10,  # 增加图像金字塔层数
                         edgeThreshold=15,  # 降低边缘阈值
                         firstLevel=0,
                         WTA_K=3,  # 改变生成描述符的点数
                         scoreType=cv2.ORB_HARRIS_SCORE,
                         patchSize=25,  # 减小patchSize
                         fastThreshold=10)  # 进一步降低FAST角点检测阈值 5
    img = enhanced_preprocess(image)
    kp, des = orb.detectAndCompute(img, None)
    return kp, des


def orb_similarity(image1, image2):
    kp1, des1 = extract_orb_features(image1)
    kp2, des2 = extract_orb_features(image2)
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
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
    square_image1, square_image2, best_rotated_image2, best_angle = find_best_rotation_angle(image_path1, image_path2)

    kp1, des1 = extract_orb_features(square_image1)
    kp2_rotated, des2_rotated = extract_orb_features(best_rotated_image2)

    similarity = orb_similarity(square_image1, best_rotated_image2)
    max_possible_matches = min(len(kp1), len(kp2_rotated))
    similarity_percentage = (similarity / max_possible_matches) * 100 if max_possible_matches > 0 else 0
    print(f"参照图像 {image_path1} 和旋转后图像的相似度: {similarity_percentage:.2f}%")

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2_rotated)
    matches = sorted(matches, key=lambda x: x.distance)

    print(f"参照图像 {image_path1} 的关键点个数: {len(kp1)}")
    print(f"旋转后图像 {image_path2} 的关键点个数: {len(kp2_rotated)}")

    img1_with_kp = cv2.drawKeypoints(np.array(square_image1), kp1, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    img2_with_kp = cv2.drawKeypoints(np.array(best_rotated_image2), kp2_rotated, None, color=(0, 255, 0), flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    for match in matches:
        img1_idx = match.queryIdx
        img2_idx = match.trainIdx
        (x1, y1) = kp1[img1_idx].pt
        (x2, y2) = kp2_rotated[img2_idx].pt
        cv2.line(img1_with_kp, (int(x1), int(y1)), (int(x2), int(y2)), (255, 0, 0), 1)

    combined_image = np.hstack((img1_with_kp, img2_with_kp))

    # plt.figure(figsize=(12, 6))
    # plt.imshow(cv2.cvtColor(combined_image, cv2.COLOR_BGR2RGB))
    # plt.title(f"参照图像 {image_path1} 和旋转后图像的关键点匹配")
    # plt.show()