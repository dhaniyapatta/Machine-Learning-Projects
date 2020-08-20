import numpy as np
import cv2 


cap = cv2.VideoCapture("lane_vgt.mp4")

while(1):

    # Take each frame
    _, frame = cap.read()

    hls=cv2.cvtColor(frame, cv2.COLOR_RGB2HLS)
    hsv=cv2.cvtColor(frame, cv2.COLOR_RGB2HSV)
    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    lower_w=np.array([43,53,159])
    higher_w=np.array([64,100,206])
    hsv_mask=cv2.inRange(hsv,lower_w,higher_w)
    masked_image=cv2.bitwise_and(hsv,hsv_mask)
   
    cv2.imshow("masked",masked_image)

  
    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()