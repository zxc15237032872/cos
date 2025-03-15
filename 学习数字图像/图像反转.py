import cv2
import numpy as np

# 读取图像
# cv2.imread() 函数用于读取指定路径的图像文件
# 第一个参数是图像文件的路径，第二个参数 cv2.IMREAD_COLOR 表示以彩色模式读取图像
image = cv2.imread('005A.png', cv2.IMREAD_COLOR)

# 检查图像是否成功读取
# 如果图像读取失败，cv2.imread() 会返回 None
if image is None:
    print("无法读取图像，请检查图像路径是否正确。")
else:
    # 进行图像反转
    # 图像反转的原理是用 255 减去图像中的每个像素值
    # 因为在8位图像中，像素值的范围是 0-255，所以用 255 减去原像素值可以得到反转后的像素值
    inverted_image = 255 - image

    # 显示原始图像和反转后的图像
    # cv2.imshow() 函数用于显示图像
    # 第一个参数是窗口的名称，第二个参数是要显示的图像
    cv2.imshow('Original Image', image)
    cv2.imshow('Inverted Image', inverted_image)

    # 等待按键事件
    # cv2.waitKey(0) 表示无限等待用户按下任意键
    cv2.waitKey(0)

    # 关闭所有打开的窗口
    # cv2.destroyAllWindows() 用于关闭所有由 OpenCV 创建的窗口
    cv2.destroyAllWindows()

    # 保存反转后的图像
    # cv2.imwrite() 函数用于将图像保存到指定路径
    # 第一个参数是保存的文件路径，第二个参数是要保存的图像
    cv2.imwrite('inverted_005A.png', inverted_image)
    print("反转后的图像已保存为 inverted_005A.png")