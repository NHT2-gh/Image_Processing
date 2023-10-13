# Name: Nguyen Huynh Truc Thanh
# Id :  N20DCCN142

import cv2

# a
J1 = cv2.imread("img/hw2/lena512color.jpg")
# b
cv2.imshow("J1", J1)
cv2.waitKey(0)
cv2.destroyAllWindows()

# c
J2 = J1.copy()
J2[:, :, 0] = J1[:, :, 2]
J2[:, :, 1] = J1[:, :, 0]
J2[:, :, 2] = J1[:, :, 1]

# d
cv2.imshow("Original Image", J1)
cv2.imshow("J2 - Swapped Image", J2)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite("img/hw2/lena512color_swapped.jpg", J2)
