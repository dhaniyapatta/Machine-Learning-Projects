import cv2 
import numpy as np




Pts = [] # for storing points
cap = cv2.VideoCapture("lane_vgt.mp4")

# :mouse callback function

def draw_roi(event, x, y, flags, param):
    img2 = img.copy()
 
    If event == cv2.EVENT_LBUTTONDOWN: # Left click, select point
        pts.append((x, y))  
 
    If event == cv2.EVENT_RBUTTONDOWN: # Right click to cancel the last selected point
        pts.pop()  
 
    If event == cv2.EVENT_MBUTTONDOWN: # 
        mask = np.zeros(img.shape, np.uint8)
        points = np.array(pts, np.int32)
        points = points.reshape((-1, 1, 2))
                             # 
                   
    mask = cv2.polylines(mask, [points], True, (255, 255, 255), 2)
    Mask2 = cv2.fillPoly(mask.copy(), [points], (255, 255, 255)) # for ROI
    Mask3 = cv2.fillPoly(mask.copy(), [points], (0, 255, 0)) # for displaying images on the desktop
             
    show_image = cv2.addWeighted(src1=img, alpha=0.8, src2=mask3, beta=0.2, gamma=0)
             
    cv2.imshow("mask", mask2)
    cv2.imshow("show_img", show_image)
             
    ROI = cv2.bitwise_and(mask2, img)
    cv2.imshow("ROI", ROI)


while (True):
    _,frame=cap.read()
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', draw_roi)
    k = cv2.waitKey(1000) & 0xFF # large wait time to remove freezing
    if k == 113 or k == 27:
        break
