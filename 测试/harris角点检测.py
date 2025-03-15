import cv2
import numpy as np
import os


def harris_corner_detection():
    image_path = 'D:/2024study/project/car/cos/image/网格图.jpg'
    with open(image_path, 'rb') as f:
        img_data = f.read()
    img = cv2.imdecode(np.frombuffer(img_data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        print(f"无法读取文件 {image_path}，可能是文件格式或其他问题。")
        return
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = np.float32(gray)

    dst = cv2.cornerHarris(gray, 2, 3, 0.04)
    dst = cv2.dilate(dst, None)
    img[dst > 0.01 * dst.max()] = [0, 0, 255]

    cv2.imshow('Harris Corners', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    harris_corner_detection()