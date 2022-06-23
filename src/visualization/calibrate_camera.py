import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt


"""
Implement the number of vertical and horizontal corners
"""
# nb_horizontal = 8
# nb_vertical = 5


# # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# objp = np.zeros((nb_horizontal*nb_vertical,3), np.float32)
# objp[:,:2] = np.mgrid[0:nb_vertical,0:nb_horizontal].T.reshape(-1,2)
# objp = objp*3.3 #33.6 multiplying with the actual size of the checker board square

# # Arrays to store object points and image points from all the images.
# objpoints = [] 
# imgpoints = [] 

images = sorted(glob.glob('data/raw/calibration/frame*.png')) #/home/kristin/bumpyProject/data/raw/calibration/frame001142.png

# assert images
# for fname in images:
#     img = cv2.imread(fname)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     ret, corners = cv2.findChessboardCorners(gray, (nb_vertical, nb_horizontal), None)

#     # If found, add object points, image points (after refining them)
#     if ret == True:
#         objpoints.append(objp)
#         imgpoints.append(corners)

#         # Draw and display the corners
# #         img = cv2.drawChessboardCorners(img, (nb_vertical,nb_horizontal), corners,ret)
# #         cv2.imshow('img',img)
# #         cv2.waitKey(500)

# # cv2.destroyAllWindows()

# #obtaining and then saving camera parameters to a file for later use
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# np.savez("calib_results.npz", ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

# print(mtx)
# # [[584.78017866   0.         740.35975462]
# #  [  0.         585.3587445  376.26495791]
# #  [  0.           0.           1.        ]]


####################################################################################################
#reading in the camera parameters from a file for testing
npzfile = np.load("calib_results.npz")
ret, mtx, dist, rvecs, tvecs = npzfile["ret"], npzfile["mtx"], npzfile["dist"], npzfile["rvecs"], npzfile["tvecs"]

#reading a test image
img = cv2.imread(images[0])
#img = cv2.imread("data/processed/imgs/frame000878.png")
h,  w = img.shape[:2] #Original: 732 , 1490
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 1, (w,h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)
# crop the image
#x, y, w, h = roi #Undistorted: 1206 , 711
#dst = dst[y:y+h, x:x+w]
cv2.imshow('original', img)
cv2.imshow('calibresult', dst)
cv2.waitKey()
#cv2.imwrite('calibresult.png', dst)
cv2.destroyAllWindows()