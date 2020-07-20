#Edge Detection And Gradient

mport cv2
import numpy as np

cap = cv2.VideoCapture(0)

while (1):

    _, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    laplacian= cv2.Laplacian(frame, cv2.CV_64F)
    sobelx = cv2.Sobel(frame, cv2.CV_64F,1,0,ksize=5)
    sobely = cv2.Sobel(frame, cv2.CV_64F,0,1,ksize=5)
    edge= cv2.Canny(frame,100,100)
    cv2.imshow("orignal",frame)
    cv2.imshow("laplacian",laplacian)
    cv2.imshow("sobelx",sobelx)
    cv2.imshow("sobely",sobely)
    cv2.imshow("edges",edge)


    k = cv2.waitKey(5) & 0xFF
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()