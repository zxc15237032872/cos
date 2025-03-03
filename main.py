import cv2

image = cv2.imread("001A.png", cv2.IMREAD_GRAYSCALE)
cv2.imshow("Original Image", image)
cv2.waitKey(0)