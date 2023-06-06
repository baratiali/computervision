import cv2
import numpy as np
import imutils as imu

cam = cv2.VideoCapture(0)
template = cv2.imread("img/face.jpg")
template = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
_, template = cv2.threshold(template, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
(tH, tW) = template.shape[:2]

while True:
    _, frame = cam.read()
    found = None
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    for scale in np.linspace(0.2, 1.0, 20)[::-1]:
        resized = imu.resize(frame_gray, width=int(frame_gray.shape[1] * scale))
        r = frame_gray.shape[1] / float(resized.shape[1])
        if resized.shape[0] < tH or resized.shape[1] < tW:
            break

        _, frame_thresh = cv2.threshold(resized, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        result = cv2.matchTemplate(frame_thresh, template, cv2.TM_CCOEFF)
        (_, maxVal, _, maxLoc) = cv2.minMaxLoc(result)
        
        if found is None or maxVal > found[0]:
            found = (maxVal, maxLoc, r)

        (_, maxLoc, r) = found
        (startX, startY) = (int(maxLoc[0] * r), int(maxLoc[1] * r))
        (endX, endY) = (int((maxLoc[0] + tW) * r), int((maxLoc[1] + tH) * r))
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 0, 255), 10)
    
    cv2.imshow("Face Detection", frame)
    if cv2.waitKey(1) == 27:
        break

cam.release()
cv2.destroyAllWindows()
