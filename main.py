import numpy as np
import cv2

font = cv2.FONT_HERSHEY_SIMPLEX

image = cv2.imread("img/coins.jpg")
blurred = cv2.GaussianBlur(image, (7, 7), 0)
filtered = cv2.pyrMeanShiftFiltering(blurred, 20, 10)
gray = cv2.cvtColor(filtered, cv2.COLOR_BGR2GRAY)

_, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, None, iterations=1)

_, labels = cv2.connectedComponents(thresh, connectivity=8, ltype=cv2.CV_32S)
mask = np.zeros(thresh.shape, np.uint8)
center = []

for label in np.unique(labels):
    if label == 0:
        continue

    labelMask = np.zeros(thresh.shape, dtype="uint8")
    labelMask[labels == label] = 255
    numPixels = cv2.countNonZero(labelMask)

    if numPixels > 300:
        M = cv2.moments(labelMask)
        cX = int(M['m10'] / M['m00'])
        cY = int(M['m01'] / M['m00'])
        center.append((cX, cY))
        mask = cv2.add(mask, labelMask)

for i, center in enumerate(center):
    cv2.circle(image, center, 30, (32, 100, 255), 2)
    cv2.putText(image, str(i + 1), center, font, 1, (0, 0, 255), 3)

cv2.imshow("Detecting Coins", image)
cv2.waitKey(0)
cv2.destroyAllWindows()
