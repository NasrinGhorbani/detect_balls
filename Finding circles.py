import cv2
import numpy as np
import matplotlib.pyplot as plt

kernel_sharp = np.array([[0, -1, 0],
	[-1, 5, -1],
	[0, -1, 0]])

# Loading image

img = cv2.imread("00tennisballs1-superJumbo.jpg")

sharp1 = cv2.filter2D(img, -1, kernel_sharp)

img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

sharp = cv2.filter2D(gray, -1, kernel_sharp)

edges = cv2.Canny(sharp, 100, 255)

hsv = cv2.cvtColor(sharp1, cv2.COLOR_BGR2HSV)


circles = cv2.HoughCircles(edges, cv2.HOUGH_GRADIENT, 3, 330, minRadius=75, maxRadius=107)
# print(f"circles on edges image: {circles}")


circles = circles[0].astype(np.uint32)

if circles is not None:
	for circle in circles:
		cv2.circle(img_rgb, (circle[0], circle[1]), circle[2], (255, 0, 0), 8)


print(f"There are {len(circles)} balls in this picture.")


lower_red = np.array([0, 70, 50])
upper_red = np.array([10, 255, 255])
mask1 = cv2.inRange(hsv, lower_red, upper_red)

lower_red = np.array([170, 70, 50])
upper_red = np.array([180, 255, 255])
mask2 = cv2.inRange(hsv, lower_red, upper_red)

mask = mask1 + mask2

mask = cv2.dilate(mask, None, iterations=2)
mask = cv2.erode(mask, None, iterations=1)

# resize = cv2.resize(mask, None, fx=0.4, fy=0.4)
# cv2.imshow('Detected Red Balls', resize)
# cv2.waitKey(0)

circles = cv2.HoughCircles(mask, cv2.HOUGH_GRADIENT, 3.1, 400, minRadius=94, maxRadius=105)
# print(f" Red circle on mask image: {circles}")

circles = circles[0].astype(np.uint32)

if circles is not None:
    for circle in circles:
        cv2.circle(img_rgb, (circle[0], circle[1]), circle[2], (0, 255, 0), 8)
        print(f"The coordinates of the red ball is ({circle[0]}, {circle[1]}, {circle[2]}).")

plt.imshow(img_rgb, cmap="gray")
plt.show()
