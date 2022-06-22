import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


"""
Implement the number of vertical and horizontal corners
"""
nb_horizontal = 8
nb_vertical = 5


# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((nb_horizontal*nb_vertical,3), np.float32)
objp[:,:2] = np.mgrid[0:nb_vertical,0:nb_horizontal].T.reshape(-1,2)
objp = objp*33.6 #multiplying with the actual size of the checker board square

# Arrays to store object points and image points from all the images.
objpoints = [] 
imgpoints = [] 

images = glob.glob('data/raw/calibration/frame*.png') #/home/kristin/bumpyProject/data/raw/calibration/frame001142.png

assert images
for fname in images:
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    ret, corners = cv2.findChessboardCorners(gray, (nb_vertical, nb_horizontal), None)

    # If found, add object points, image points (after refining them)
    if ret == True:
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
#         img = cv2.drawChessboardCorners(img, (nb_vertical,nb_horizontal), corners,ret)
#         cv2.imshow('img',img)
#         cv2.waitKey(500)

# cv2.destroyAllWindows()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# print(ret)
print(mtx)
# [[584.78019224   0.         740.35969583]
#  [  0.         585.35875633 376.26496857]
#  [  0.           0.           1.        ]]
# print(dist)
# print(rvecs)
# print(tvecs)

img = cv2.imread(images[0])
h,  w = img.shape[:2]
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))
