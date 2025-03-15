import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False
import numpy as np
from PIL import Image
from skimage.metrics import structural_similarity as ssim
import cv2
import torch
import torchvision.models as models
import torchvision.transforms as transforms


def convert_to_square(image, size=70):
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
        resized_image = image.resize((new_width, image.height))
    else:
        new_height = int(image.height * scale_ratio)
        resized_image = image.resize((image.width, new_height))
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
    final_image = cropped_image.resize((size, size))
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


# 加载预训练的孪生网络模型
class SiameseNetwork(torch.nn.Module):
    def __init__(self):
        super(SiameseNetwork, self).__init__()
        self.cnn = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        num_ftrs = self.cnn.fc.in_features
        self.cnn.fc = torch.nn.Identity()

    def forward_once(self, x):
        output = self.cnn(x)
        return output

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


def extract_features(model, image):
    transform = transforms.Compose([
        transforms.Resize((90, 90)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    img_tensor = transform(image).unsqueeze(0)
    # 创建一个虚拟的输入
    dummy_tensor = torch.zeros_like(img_tensor)
    with torch.no_grad():
        features1, _ = model(img_tensor, dummy_tensor)
    return features1.squeeze().numpy()


def cosine_similarity(feature1, feature2):
    dot_product = np.dot(feature1, feature2)
    norm_feature1 = np.linalg.norm(feature1)
    norm_feature2 = np.linalg.norm(feature2)
    similarity = dot_product / (norm_feature1 * norm_feature2)
    return similarity


image_pairs = [
    ('006A.png', '006A1.png'),
    ('004A.png', '004A1.png'),
    ('005A.png', '005B.png'),
    ('006A.png', '004A1.png'),
    ('004A.png', '005A.png'),
    ('005A.png', '006A.png')
]

model = SiameseNetwork()
model.eval()

for pair in image_pairs:
    image_path1, image_path2 = pair
    square_image1, square_image2, best_rotated_image2, best_angle = find_best_rotation_angle(image_path1, image_path2)
    feature1 = extract_features(model, square_image1)
    feature2 = extract_features(model, square_image2)
    feature2_rotated = extract_features(model, best_rotated_image2)
    similarity_before_rotation = cosine_similarity(feature1, feature2)
    similarity_after_rotation = cosine_similarity(feature1, feature2_rotated)

    print(f"旋转前 {image_path1} 和 {image_path2} 的相似度: {similarity_before_rotation}")
    print(f"旋转 {best_angle} 度后 {image_path1} 和 {image_path2} 的相似度: {similarity_after_rotation}")

    # plt.figure(figsize=(12, 4))
    # plt.subplot(1, 3, 1)
    # plt.imshow(np.array(square_image1))
    # plt.title(f'图片 {image_path1}')
    # plt.subplot(1, 3, 2)
    # plt.imshow(np.array(square_image2))
    # plt.title(f'图片 {image_path2}')
    # plt.subplot(1, 3, 3)
    # plt.imshow(np.array(best_rotated_image2))
    # plt.title(f'图片 {image_path2} 旋转 {best_angle} 度后')
    # plt.show()
