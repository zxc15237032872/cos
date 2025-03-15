import cv2
import numpy as np


def average_hash(image, hash_size=8):
    # 调整图像大小
    resized = cv2.resize(image, (hash_size * 2, hash_size * 2), interpolation=cv2.INTER_AREA)
    # 转换为灰度图像
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    # 计算图像的均值
    mean = np.mean(gray)
    # 生成哈希值
    hash_value = np.where(gray > mean, 1, 0)
    return hash_value.flatten()


def hamming_distance(hash1, hash2):
    # 计算汉明距离
    return np.count_nonzero(hash1 != hash2)


# 读取图片
image_path1 = '004A.png'  # 替换为你的第一张轮毂图片路径
image_path2 = '004A1.png'  # 替换为你的第二张轮毂图片路径
image1 = cv2.imread(image_path1)
image2 = cv2.imread(image_path2)

# 计算哈希值
hash1 = average_hash(image1)
hash2 = average_hash(image2)

# 计算汉明距离
distance = hamming_distance(hash1, hash2)
# 可以根据汉明距离计算相似度，这里简单以 1 - (distance / (len(hash1))) 计算相似度
similarity = 1 - (distance / len(hash1))
print(f"两张图片的相似度: {similarity}")
