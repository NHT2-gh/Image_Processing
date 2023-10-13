# Name: Nguyen Huynh Truc Thanh
# Id :  N20DCCN142

import cv2

help(cv2.imread)
help(cv2.imwrite)

J1 = cv2.imread("img/hw2/lena512color.jpg")

if J1 is not None:

    J2 = 255 - J1
    cv2.imshow("Original Image", J1)
    cv2.imshow("J2 (Photographic Negative)", J2)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    cv2.imwrite("img/hw2/lena_negative.jpg", J2)
else:
    print("Image not found")