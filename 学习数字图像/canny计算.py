import cv2
import numpy as np
import cv2
import numpy as np
import matplotlib.pyplot as plt
import cv2
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt

# 读取图像
image = cv2.imread('005A.png')

# 检查图像是否成功读取
if image is None:
    print("无法读取图像，请检查图像路径和文件名是否正确。")
    exit(1)

# 转换为灰度图像
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# 高斯模糊以减少噪声
blurred = cv2.GaussianBlur(gray, (5, 5), 0)

# 使用Canny算子检测边缘
edges = cv2.Canny(blurred, 50, 150)

# 霍夫直线检测
lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=10)

# 统计辐条数量
spoke_count = 0
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        # 这里可以根据直线的角度等信息进一步筛选出属于辐条的直线
        # 简单示例：假设辐条直线有一定的角度范围
        angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
        if -45 < angle < 45 or 135 < angle < 180 or -180 < angle < -135:
            spoke_count += 1

print(f"轮毂的辐条数量: {spoke_count}")

# 可视化结果
plt.figure(figsize=(12, 6))

plt.subplot(121)
plt.imshow(gray, cmap='gray')
plt.title('灰度图像')
plt.axis('off')

plt.subplot(122)
plt.imshow(edges, cmap='gray')
plt.title('Canny边缘检测结果')
plt.axis('off')

plt.show()

# 在原图上绘制检测到的直线（可选）
if lines is not None:
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(image, (x1, y1), (x2, y2), (0, 0, 255), 2)

    cv2.imshow('Detected Lines', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()