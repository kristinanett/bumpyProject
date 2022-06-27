import numpy as np
import cv2
import glob

# nb_horizontal = 9
# nb_vertical = 6
# criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# # prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
# objp = np.zeros((nb_horizontal*nb_vertical,3), np.float32)
# objp[:,:2] = np.mgrid[0:nb_vertical,0:nb_horizontal].T.reshape(-1,2)
# objp = objp*40 #multiplying with the actual size of the checker board square

# # Arrays to store object points and image points from all the images.
# objpoints = [] 
# imgpoints = [] 

# #change path to images calibration images
images = sorted(glob.glob('data/raw/calibration2/cam_2022-06-27-13-04-13_0/frame*.png'))

# assert images
# for fname in images:
#     img = cv2.imread(fname)
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

#     ret, corners = cv2.findChessboardCorners(gray, (nb_vertical, nb_horizontal), None)

#     # If found, add object points, image points (after refining them)
#     if ret == True:
#         objpoints.append(objp)

#         corners2 = cv2.cornerSubPix(gray,corners, (11,11), (-1,-1), criteria)
#         imgpoints.append(corners2)

#         #Draw and display the corners
#         img = cv2.drawChessboardCorners(img, (nb_vertical,nb_horizontal), corners2, ret)
#         cv2.imshow('img',img)
#         cv2.waitKey(500)

# cv2.destroyAllWindows()

# #obtaining and then saving camera parameters to a file for later use
# ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)
# np.savez("calib_results.npz", ret=ret, mtx=mtx, dist=dist, rvecs=rvecs, tvecs=tvecs)

#print(mtx)
# [[583.91527331   0.         754.24062037]
#  [  0.         583.0798202  371.36480426]
#  [  0.           0.           1.        ]]

####################################################################################################
#reading in the camera parameters from a file for testing
npzfile = np.load("calib_results3.npz")
ret, mtx, dist, rvecs, tvecs = npzfile["ret"], npzfile["mtx"], npzfile["dist"], npzfile["rvecs"], npzfile["tvecs"]

#reading a test image
img = cv2.imread(images[90])
#img = cv2.imread("data/processed/imgs/frame000878.png")
h,  w = img.shape[:2] #original image size: 732 , 1490
newcameramtx, roi = cv2.getOptimalNewCameraMatrix(mtx, dist, (w,h), 0, (w,h))

# undistort
dst = cv2.undistort(img, mtx, dist, None, newcameramtx)

cv2.imshow('original', img)
cv2.imshow('calibresult', dst)
cv2.waitKey()
cv2.destroyAllWindows()