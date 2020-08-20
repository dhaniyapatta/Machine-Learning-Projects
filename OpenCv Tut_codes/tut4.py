import numpy as np
import  cv2

img=cv2.imread('hahacat.jpg',cv2.IMREAD_COLOR)



# px= img[55,55]
#
# print(px)
#
# img[55,55]=[255,255,255]
#
# px= img[55,55]
#
# print(px)
#

img[100:150, 100:150]=[255,255,255]

roi=img[37:111,107:194]  # 111-73=74  194-107=87

img[0:74,0:87]=roi


cv2.imshow("image",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
