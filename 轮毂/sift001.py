import cv2
import numpy as np
from skimage.transform import resize
import matplotlib.pyplot as plt
import os
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

def sift_feature_matching(image1, image2):

    sift = cv2.SIFT_create(nfeatures = 1000)
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)

    if des1 is None or des2 is None:
        return 0

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k = 2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append([m])

    similarity = len(good) / len(matches) if len(matches) > 0 else 0
    return similarity


def orb_feature_matching(image1, image2):
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image1, None)
    kp2, des2 = orb.detectAndCompute(image2, None)

    if des1 is None or des2 is None:
        return 0

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = bf.match(des1, des2)
    matches = sorted(matches, key = lambda x: x.distance)

    similarity = len(matches) / min(len(des1), len(des2)) if min(len(des1), len(des2)) > 0 else 0
    return similarity


def combined_feature_matching(image1, image2):
    sift_sim = sift_feature_matching(image1, image2)
    orb_sim = orb_feature_matching(image1, image2)
    combined_sim = 0.6 * sift_sim + 0.4 * orb_sim
    return combined_sim


def preprocess_image(image_path):
    try:
        with open(image_path, 'rb') as f:
            image_data = f.read()
        nparr = np.frombuffer(image_data, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
        if image is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")

        target_size = (256, 256)
        image = resize(image, target_size, anti_aliasing = True)
        image = np.array(image * 255, dtype = np.uint8)

        image = cv2.equalizeHist(image)
        image = cv2.GaussianBlur(image, (5, 5), 0)
    except Exception as e:
        raise FileNotFoundError(f"Error reading image {image_path}: {e}")
    return image


def calculate_similarity(image_path1, image_path2):
    image1 = preprocess_image(image_path1)
    image2 = preprocess_image(image_path2)

    similarity = combined_feature_matching(image1, image2)
    return similarity


def visualize_images(images, titles):
    num_images = len(images)
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(1, num_images, i + 1)
        plt.imshow(images[i], cmap='gray')
        plt.title(titles[i])
        plt.axis('off')
    plt.show()


def draw_matches(image1, image2, kp1, kp2, matches):
    height1, width1 = image1.shape
    height2, width2 = image2.shape
    new_image = np.zeros((max(height1, height2), width1 + width2, 3), dtype=np.uint8)
    new_image[:height1, :width1] = cv2.cvtColor(image1, cv2.COLOR_GRAY2BGR)
    new_image[:height2, width1:] = cv2.cvtColor(image2, cv2.COLOR_GRAY2BGR)

    for match in matches:
        idx1 = match.queryIdx
        idx2 = match.trainIdx
        (x1, y1) = kp1[idx1].pt
        (x2, y2) = kp2[idx2].pt
        x2 += width1
        color = tuple(np.random.randint(0, 255, size=3).tolist())
        cv2.line(new_image, (int(x1), int(y1)), (int(x2), int(y2)), color, 1)

    return new_image


def visualize_matches(image1, image2):
    sift = cv2.SIFT_create(nfeatures = 1000)
    kp1, des1 = sift.detectAndCompute(image1, None)
    kp2, des2 = sift.detectAndCompute(image2, None)

    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k = 2)

    good = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            good.append(m)

    match_image = draw_matches(image1, image2, kp1, kp2, good)
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(match_image, cv2.COLOR_BGR2RGB))
    plt.title('Matched Features')
    plt.axis('off')
    plt.show()


image_files = ["轮毂001B1.png", "005B.png", "005A.png"]
num_images = len(image_files)

preprocessed_images = []
for file in image_files:
    preprocessed_image = preprocess_image(file)
    preprocessed_images.append(preprocessed_image)

visualize_images(preprocessed_images, image_files)

for i in range(num_images):
    for j in range(i + 1, num_images):
        img1 = preprocessed_images[i]
        img2 = preprocessed_images[j]
        visualize_matches(img1, img2)
        similarity = calculate_similarity(image_files[i], image_files[j])
        print(f"Similarity between {image_files[i]} and {image_files[j]} is: {similarity}")
