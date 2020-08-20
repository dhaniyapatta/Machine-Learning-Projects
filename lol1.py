import numpy as np
import cv2 


cap = cv2.VideoCapture("lane_vgt.mp4")
def callback(x):
	pass 

ilowH = 43
ihighH = 67

ilowS = 52
ihighS = 103
ilowV = 156
ihighV = 220

kernel_size=5 
cv2.namedWindow('image')


# create trackbars for color change
cv2.createTrackbar('lowH','image',ilowH,179,callback)
cv2.createTrackbar('highH','image',ihighH,179,callback)

cv2.createTrackbar('lowS','image',ilowS,255,callback)
cv2.createTrackbar('highS','image',ihighS,255,callback)

cv2.createTrackbar('lowV','image',ilowV,255,callback)
cv2.createTrackbar('highV','image',ihighV,255,callback)


while(True):
    # grab the frame
    ret, frame = cap.read()

    # get trackbar positions
    ilowH = cv2.getTrackbarPos('lowH', 'image')
    ihighH = cv2.getTrackbarPos('highH', 'image')
    ilowS = cv2.getTrackbarPos('lowS', 'image')
    ihighS = cv2.getTrackbarPos('highS', 'image')
    ilowV = cv2.getTrackbarPos('lowV', 'image')
    ihighV = cv2.getTrackbarPos('highV', 'image')

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsl = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    lower_hsv = np.array([ilowH, ilowS, ilowV])
    higher_hsv = np.array([ihighH, ihighS, ihighV])
    mask = cv2.inRange(hsv, lower_hsv, higher_hsv)

    masked_img= cv2.bitwise_and(frame, frame, mask=mask)


    def gauss_blur(image,kernel):
        return cv2.GaussianBlur(image,(kernel,kernel),0)
    
    def closing(image,kernel):
        k=np.ones((kernel,kernel),np.uint8)
        return cv2.morphologyEx(image,cv2.MORPH_CLOSE,k)

    def canny(image):
        low_thresh=50
        high_thresh=150
        return cv2.Canny(image,low_thresh,high_thresh)


        
    closing=closing(masked_img,kernel_size)
    # gauss=gauss_blur(closing,kernel_size)
    # canny=canny(gauss)

    # show thresholded image
    cv2.imshow('image', masked_img)
    cv2.imshow("closing",closing)
    # cv2.imshow("gauss",gauss)
    # cv2.imshow("canny",canny)






    











    k = cv2.waitKey(1000) & 0xFF # large wait time to remove freezing
    if k == 113 or k == 27:
        break