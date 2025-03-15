import numpy as np
import cv2

# 假设 histogram_matching 函数已经定义
def histogram_matching(source, reference):
    # 这里可以实现直方图匹配的具体逻辑
    # 为了简化，我们可以直接返回源图像
    return source

def preprocess_image(image):
    # 假设参考图像是一个均匀分布的图像
    reference = np.random.randint(0, 256, size=image.shape, dtype=np.uint8)
    enhanced_img = histogram_matching(image, reference).astype(np.uint8)

    # 多次锐化以增强锐化效果
    for _ in range(3):
        kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
        enhanced_img = cv2.filter2D(enhanced_img, -1, kernel)

    # 增加明暗对比
    alpha = 2  # 对比度增强因子
    enhanced_img = cv2.convertScaleAbs(enhanced_img, alpha=alpha, beta=0)

    # 增加饱和度（在 HSV 颜色空间中操作）
    hsv = cv2.cvtColor(enhanced_img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = np.clip(s * 3, 0, 255).astype(hsv.dtype)
    hsv = cv2.merge([h, s, v])
    enhanced_img = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return enhanced_img

def process_image(image):
    # 预处理图像
    preprocessed_image = preprocess_image(image)
    img_array = preprocessed_image

    # 找到图像的中心
    height, width, _ = img_array.shape
    center_x = width // 2
    center_y = height // 2

    # 遍历图像
    for y in range(height):
        for x in range(width):
            # 找到对称点的坐标
            symmetric_x = 2 * center_x - x
            symmetric_y = 2 * center_y - y

            # 检查对称点是否在图像范围内
            if 0 <= symmetric_x < width and 0 <= symmetric_y < height:
                # 获取当前像素和对称像素的颜色值
                current_pixel = img_array[y, x]
                symmetric_pixel = img_array[symmetric_y, symmetric_x]

                # 计算颜色差异
                color_difference = np.linalg.norm(current_pixel - symmetric_pixel)

                # 判断像素是否对称
                if color_difference > 50:  # 可以根据实际情况调整阈值
                    # 不对称像素，判断其是否应该存在
                    # 这里简单假设如果对称点颜色较亮，则认为该像素不应该存在
                    if np.mean(symmetric_pixel) > np.mean(current_pixel):
                        # 不存在，将像素变为黑色
                        img_array[y, x] = [0, 0, 0]
                    else:
                        # 存在，将像素变为红色
                        img_array[y, x] = [0, 0, 255]

    return img_array

# 示例用法
if __name__ == "__main__":
    # 打开图像
    image = cv2.imread("005A.png", cv2.IMREAD_UNCHANGED)
    if image.shape[2] == 4:  # 如果是 RGBA 图像
        image = image[:, :, :3]  # 去掉透明度通道
    # 调整图像大小
    resized_image = cv2.resize(image, (224, 224))

    # 处理图像
    processed_image = process_image(resized_image)

    # 保存处理后的图像
    cv2.imwrite("processed_image.jpg", processed_image)