import numpy as np
import cv2
from collections import deque

select = False
roi_pts = []
roiHist = None
pts = deque(maxlen=20)


def selectROI(event, x, y, flag, param):
    global s_frame, select, roi_pts, roiHist, clone
    if select == True and event == cv2.EVENT_LBUTTONDOWN and len(roi_pts) < 4:
        roi_pts.append((x, y))
        cv2.circle(s_frame, (x, y), 10, (0, 255, 0), -1)

        if len(roi_pts) == 4:
            (tl, tr, br, bl) = roi_pts
            roi = clone[tl[1]: br[1], tl[0]:tr[0]]
            roi_hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
            roiHist = cv2.calcHist([roi_hsv], [0, 1], None, [180, 256], [0, 180, 0, 256])
            roiHist = cv2.normalize(roiHist, roiHist, 0, 255, cv2.NORM_MINMAX)


def object_detect(roiHist, frame):
    frame_hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.calcBackProject([frame_hsv], [0, 1], roiHist, [0, 180, 0, 256], 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    cv2.filter2D(mask, -1, kernel, mask)
    return mask


cam = cv2.VideoCapture(0)
cv2.namedWindow("Tracking")
cv2.setMouseCallback("Tracking", selectROI)

while True:
    _, frame = cam.read()
    clone = frame.copy()
    key = cv2.waitKey(1)

    if key == 27:
        break
    elif key == ord("s"):
        select = not select
        if select:
            s_frame = frame
            clone = frame.copy()
            roi_pts = []

    if select:
        cv2.imshow("Tracking", s_frame)
    else:
        if len(roi_pts) == 4:
            mask = object_detect(roiHist, frame)
            M = cv2.moments(mask)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.circle(frame, (cX, cY), 15, (0, 0, 255), -1)

            pts.appendleft((cX, cY))
            for i in range(1, len(pts)):
                if pts[i - 1] is None or pts[i] is None:
                    continue
                cv2.line(frame, pts[i - 1], pts[i], (255, 0, 0), 5)

            cv2.imshow("'Mask", mask)
    cv2.imshow("Tracking", frame)

cam.release()
cv2.destroyAllWindows()
