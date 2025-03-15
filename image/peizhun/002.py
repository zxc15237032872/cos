import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置为支持中文的字体，如黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 读取图片
def read_image(image_path):
    image = cv2.imread(image_path)
    return image

# 找到图片中的车辆
def find_car(image):
    # 转换为灰度图
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # 高斯模糊
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # 边缘检测
    edges = cv2.Canny(blurred, 50, 150)
    # 查找轮廓
    contours, _ = cv2.findContours(edges.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # 找到最大的轮廓（假设为车辆）
    if len(contours) > 0:
        car_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(car_contour)
        car_image = image[y:y + h, x:x + w]
        return car_image
    return image

# 图像配准
def register_images(img1, img2):
    # 初始化SIFT检测器
    sift = cv2.SIFT_create()
    # 检测关键点和描述符
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    # 使用FLANN匹配器进行特征匹配
    FLANN_INDEX_KDTREE = 1
    index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)
    matches = flann.knnMatch(des1, des2, k=2)
    # 筛选好的匹配点
    good_matches = []
    for m, n in matches:
        if m.distance < 0.7 * n.distance:
            good_matches.append(m)
    # 获取匹配点的坐标
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
    # 计算仿射变换矩阵
    M, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    # 应用仿射变换
    h, w = img1.shape[:2]
    registered_img = cv2.warpAffine(img2, M, (w, h))
    return registered_img

# 主函数
def main():
    # 输入图片路径
    image_path1 = 'che001.png'
    image_path2 = 'che002.png'
    # 读取图片
    img1 = read_image(image_path1)
    img2 = read_image(image_path2)
    # 找到图片中的车辆
    car1 = find_car(img1)
    car2 = find_car(img2)
    # 图像配准
    registered_car2 = register_images(car1, car2)
    # 绘制三张图
    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    plt.imshow(cv2.cvtColor(car1, cv2.COLOR_BGR2RGB))
    plt.title('Car A')
    plt.axis('off')
    plt.subplot(132)
    plt.imshow(cv2.cvtColor(car2, cv2.COLOR_BGR2RGB))
    plt.title('Car B (Original)')
    plt.axis('off')
    plt.subplot(133)
    plt.imshow(cv2.cvtColor(registered_car2, cv2.COLOR_BGR2RGB))
    plt.title('Car B (Registered)')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    main()