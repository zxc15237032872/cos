import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.transform import rotate
from skimage.color import rgb2gray


def preprocess_image(image_path):
    image = cv2.imread(image_path)
    gray_image = rgb2gray(image)
    return gray_image


def enhance_illumination(image):
    # 使用直方图均衡化增强光照
    if len(image.shape) == 2:
        equalized_image = cv2.equalizeHist(np.uint8(image * 255))
        return equalized_image / 255.0
    else:
        ycrcb = cv2.cvtColor(np.uint8(image * 255), cv2.COLOR_RGB2YCrCb)
        channels = cv2.split(ycrcb)
        channels[0] = cv2.equalizeHist(channels[0])
        equalized_ycrcb = cv2.merge(channels)
        equalized_image = cv2.cvtColor(equalized_ycrcb, cv2.COLOR_YCrCb2RGB)
        return equalized_image / 255.0


def find_best_rotation(image1, image2):
    max_similarity = -1
    best_angle = 0
    for angle in range(0, 360, 1):
        rotated_image2 = rotate(image2, angle)
        similarity = ssim(image1, rotated_image2, multichannel=False)
        if similarity > max_similarity:
            max_similarity = similarity
            best_angle = angle
    return best_angle


def calculate_similarity(image1, image2):
    similarity = ssim(image1, image2, multichannel=False)
    return similarity


image_pairs = [
    ('006A.png', '006A1.png'),
    ('004A.png', '004A1.png'),
    ('005A.png', '005B.png'),
    ('006A.png', '004A1.png'),
    ('004A.png', '005A.png'),
    ('005A.png', '006A.png')
]

for pair in image_pairs:
    image1_path, image2_path = pair
    image1 = preprocess_image(image1_path)
    image2 = preprocess_image(image2_path)

    enhanced_image1 = enhance_illumination(image1)
    enhanced_image2 = enhance_illumination(image2)

    best_angle = find_best_rotation(enhanced_image1, enhanced_image2)
    rotated_image2 = rotate(enhanced_image2, best_angle)

    similarity = calculate_similarity(enhanced_image1, rotated_image2)

    print(f"图片 {image1_path}、{image2_path} 以及旋转后的 {image2_path} 的相似度: {similarity}")

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(enhanced_image1, cmap='gray')
    plt.title(f'图片 {image1_path}')
    plt.subplot(1, 3, 2)
    plt.imshow(enhanced_image2, cmap='gray')
    plt.title(f'图片 {image2_path}')
    plt.subplot(1, 3, 3)
    plt.imshow(rotated_image2, cmap='gray')
    plt.title(f'旋转后的 {image2_path}')
    plt.show()