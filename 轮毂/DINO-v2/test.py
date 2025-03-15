import cv2

# 读取图片
image = cv2.imread('005A.png')

# 检查图片是否成功读取
if image is None:
    print("无法读取图片，请检查图片路径和文件名是否正确。")
else:
    # 将彩色图像转换为灰度图像
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 显示原始图像
    cv2.imshow('Original Image', image)

    # 显示灰度图像
    cv2.imshow('Gray Image', gray_image)

    # 等待按键事件，按任意键关闭窗口
    cv2.waitKey(0)

    # 关闭所有 OpenCV 窗口
    cv2.destroyAllWindows()

    # 保存灰度图像
    cv2.imwrite('005B_gray.png', gray_image)
    print("灰度图像已保存为 005B_gray.png")