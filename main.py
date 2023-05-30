import cv2
import time

image = cv2.imread("image.jpg")
clone1 = image.copy()
clone2 = clone1.copy()

(imgW, imgH) = (clone1.shape[1], clone1.shape[0])
(winW, winH) = (100, 100)
stepSize = 50


while (True):
    
    clone1 = image.copy()
    (imgW, imgH) = (clone1.shape[1], clone1.shape[0])
    
    
    for i in range(1, 4):
        for y in range(0, imgH, stepSize):
            for x in range(0, imgW, stepSize):
                if x + winW >= imgW or y + winH >= imgH:
                    continue

                clone2 = clone1.copy()
                cv2.rectangle(clone2, (x, y), (x + winW, y + winH), (255, 0, 0))

                cv2.imshow("checking", clone2)
                cv2.waitKey(1)
                time.sleep(0.025)

        key = cv2.waitKey(1)
        if key == 27:
            break
        if key == 27:
            break

        clone1 = cv2.pyrDown(clone1)
        (imgW, imgH) = (clone1.shape[1], clone1.shape[0])

    if key == 27:
        break

cv2.destroyAllWindows()
