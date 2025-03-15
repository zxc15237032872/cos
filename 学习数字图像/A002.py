import cv2
import numpy as np
import matplotlib.pyplot as plt


def preprocess_image():
    # 读取图像
    image = cv2.imread('007A.png', 0)
    if image is None:
        raise FileNotFoundError('无法读取图像 007A.png')

    # 傅里叶变换
    f = np.fft.fft2(image)
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))

    # 低通滤波
    rows, cols = image.shape
    crow, ccol = rows // 2, cols // 2
    mask = np.ones((rows, cols), np.uint8)
    r = 50
    center = [crow, ccol]
    x, y = np.ogrid[:rows, :cols]
    mask_area = (x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r
    mask[mask_area] = 0
    fshift = fshift * mask
    magnitude_spectrum_filtered = 20 * np.log(np.abs(fshift))

    # 逆傅里叶变换
    f_ishift = np.fft.ifftshift(fshift)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)

    # 显示傅里叶变换相关图像
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1), plt.imshow(image, cmap='gray')
    plt.title('Original Image'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 2), plt.imshow(magnitude_spectrum, cmap='gray')
    plt.title('Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.subplot(1, 3, 3), plt.imshow(magnitude_spectrum_filtered, cmap='gray')
    plt.title('Filtered Magnitude Spectrum'), plt.xticks([]), plt.yticks([])
    plt.show()

    return np.uint8(img_back)


result_image = preprocess_image()
cv2.imshow('Processed Image', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()