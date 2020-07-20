import cv2
import numpy as np

cap= cv2.VideoCapture(0)

while True:
    _, frame = cap.read()
    hsv = cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    #Set hue sat value
    lower_red = np.array([100,100,100])
    upper_red = np.array([255,200,250])


    mask= cv2.inRange(hsv,lower_red, upper_red)    # if not in the range=0, else 1
    res = cv2.bitwise_and(frame, frame, mask = mask) # show frame wherever mask is 1 in the
                                                     #region

    kernel= np.ones((15,15),np.float32)/255
    smoothed= cv2.filter2D(res,-1,kernel)

    blur = cv2.GaussianBlur(res,(15,15),0)

    median = cv2.medianBlur(res,15)

    bilareral_ cv2.bilateralFilter(res,15,75,75)

    cv2.imshow('frame',frame)
    #cv2.imshow('mask',mask)
    cv2.imshow('res',res)
    cv2.imshow('blur',blur)
    cv2.imshow('median',median)
    cv2.imshow('smoot',smoothed)

    k=cv2.waitKey(5) & 0xFF

    if k==27:
        break

cv2.destroyAllWindows()
cap.release()





