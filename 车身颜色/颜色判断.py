import cv2
import os
import numpy as np
import warnings
from IPython.display import display, HTML

# 过滤 OpenCV 的警告信息
warnings.filterwarnings("ignore", category=UserWarning, module="cv2")

def get_dominant_color(combined_image):
    """
    获取图像的主色调
    :param combined_image: 输入的组合图像
    :return: 主色调的 BGR 值
    """
    # 排除黑色背景部分
    non_black_pixels = combined_image[np.any(combined_image != [0, 0, 0], axis=-1)]
    if non_black_pixels.size == 0:
        return np.array([0, 0, 0])  # 如果没有非黑色像素，主色调设为黑色

    # 对非黑色像素进行 K-means 聚类，分为 5 类
    pixels = np.float32(non_black_pixels)
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 200, 0.1)
    _, labels, centers = cv2.kmeans(pixels, 5, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)

    # 统计每个聚类的像素数量
    unique_labels, counts = np.unique(labels, return_counts=True)
    # 找到像素数量最多的聚类索引
    dominant_label = unique_labels[np.argmax(counts)]
    # 获取该聚类的中心颜色作为主色调
    dominant = np.uint8(centers[dominant_label])

    return dominant

def compare_colors(color1, color2, threshold=30):
    """
    比较两个颜色是否相似
    :param color1: 第一个颜色的 BGR 值
    :param color2: 第二个颜色的 BGR 值
    :param threshold: 颜色差异阈值
    :return: True 表示颜色相似，False 表示颜色差异较大
    """
    distance = np.linalg.norm(np.array(color1) - np.array(color2))
    return distance < threshold

def read_image_with_chinese_path(image_path):
    """
    读取含有中文路径的图片
    """
    try:
        with open(image_path, 'rb') as f:
            img_bytes = np.asarray(bytearray(f.read()), dtype=np.uint8)
            img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)
            return img
    except Exception as e:
        return None

def get_folder_dominant_color(folder_path):
    """
    计算文件夹中所有图片混合后的主色调
    :param folder_path: 文件夹路径
    :return: 主色调的 BGR 值
    """
    images = []
    invalid_files = []
    if not os.path.exists(folder_path):
        print(f"警告：文件夹 {folder_path} 不存在，跳过。")
        return None, invalid_files
    image_files = [f for f in os.listdir(folder_path) if f.endswith('.png')]
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        if not os.path.exists(image_path):
            invalid_files.append((image_path, "文件不存在"))
            continue
        image = read_image_with_chinese_path(image_path)
        if image is None:
            invalid_files.append((image_path, "无法读取文件"))
            continue
        images.append(image)
    if not images:
        if invalid_files:
            print(f"警告：文件夹 {folder_path} 中的无效 PNG 文件：")
            for file, reason in invalid_files:
                print(f"  - {file}: {reason}")
        return None, invalid_files
    combined_image = np.vstack(images)
    return get_dominant_color(combined_image), invalid_files

def visualize_dominant_colors(before_color, after_color, sub_folder):
    """
    在 Jupyter Notebook 中可视化主色调
    :param before_color: 之前的主色调 (RGB)
    :param after_color: 之后的主色调 (RGB)
    :param sub_folder: 子文件夹名称
    """
    before_color_hex = '#{:02x}{:02x}{:02x}'.format(*before_color)
    after_color_hex = '#{:02x}{:02x}{:02x}'.format(*after_color)

    html = f"""
    <div>
        <span style="display: inline-block; width: 50px; height: 50px; background-color: {before_color_hex}; margin-right: 10px;"></span>
        <span style="display: inline-block; width: 50px; height: 50px; background-color: {after_color_hex}; margin-right: 10px;"></span>
        <span>Before {sub_folder}: {before_color}, After {sub_folder}: {after_color}</span>
    </div>
    """
    display(HTML(html))

def process_folder_pair(before_folder, after_folder, sub_folder):
    """
    处理一对同名的子文件夹，判断其中图片的主色调是否改变
    :param before_folder: 之前文件夹中的子文件夹路径
    :param after_folder: 之后文件夹中的子文件夹路径
    :param sub_folder: 子文件夹名称
    :return: True 表示颜色有改变，False 表示颜色无改变
    """
    before_dominant_color, before_invalid = get_folder_dominant_color(before_folder)
    after_dominant_color, after_invalid = get_folder_dominant_color(after_folder)

    if before_dominant_color is None or after_dominant_color is None:
        print(f"警告：{before_folder} 或 {after_folder} 中无有效 PNG 图片，跳过比较。")
        return False

    before_dominant_color_rgb = cv2.cvtColor(np.uint8([[before_dominant_color]]), cv2.COLOR_BGR2RGB)[0][0]
    after_dominant_color_rgb = cv2.cvtColor(np.uint8([[after_dominant_color]]), cv2.COLOR_BGR2RGB)[0][0]

    print(f"文件夹 {sub_folder} 之前的主色调 (RGB): {before_dominant_color_rgb}")
    print(f"文件夹 {sub_folder} 之后的主色调 (RGB): {after_dominant_color_rgb}")

    # 可视化主色调
    visualize_dominant_colors(before_dominant_color_rgb, after_dominant_color_rgb, sub_folder)

    return not compare_colors(before_dominant_color, after_dominant_color)

def process_top_folders(before_top_folder, after_top_folder):
    """
    处理之前和之后的顶级文件夹，遍历其中的同名子文件夹进行比较
    :param before_top_folder: 之前的顶级文件夹路径
    :param after_top_folder: 之后的顶级文件夹路径
    """
    if not os.path.exists(before_top_folder) or not os.path.exists(after_top_folder):
        print("警告：之前或之后的顶级文件夹不存在，请检查路径。")
        return
    before_sub_folders = [f for f in os.listdir(before_top_folder) if os.path.isdir(os.path.join(before_top_folder, f))]
    after_sub_folders = [f for f in os.listdir(after_top_folder) if os.path.isdir(os.path.join(after_top_folder, f))]

    common_sub_folders = set(before_sub_folders).intersection(set(after_sub_folders))
    for sub_folder in sorted(common_sub_folders):
        before_sub_folder_path = os.path.join(before_top_folder, sub_folder)
        after_sub_folder_path = os.path.join(after_top_folder, sub_folder)

        color_changed = process_folder_pair(before_sub_folder_path, after_sub_folder_path, sub_folder)
        if color_changed:
            print(f"文件夹 {sub_folder} 中的图片主色调发生了改变。")
        else:
            print(f"文件夹 {sub_folder} 中的图片主色调未发生改变。")

# 示例调用
before_top_folder = "之前"
after_top_folder = "之后"
process_top_folders(before_top_folder, after_top_folder)