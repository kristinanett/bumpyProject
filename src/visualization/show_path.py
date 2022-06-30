import numpy as np
import cv2
import pandas as pd
debug_dt = 0.8 #0.25 in the paper #0.6 seems pretty good for me
x = 0.38 #distance from front wheel to robot centre

#function from paper
def commands_to_positions(linvel, angvel):
    N = len(linvel) #8
    all_angles = [np.zeros(N)]
    all_positions = [np.zeros((N, 2))] #8, 2
    for linvel_i, angvel_i in zip(linvel.T, angvel.T):
        angle_i = all_angles[-1] + debug_dt * angvel_i
        position_i = all_positions[-1] + debug_dt * linvel_i[..., np.newaxis] * np.stack([np.cos(angle_i), np.sin(angle_i)], axis=1)

        all_angles.append(angle_i)
        all_positions.append(position_i)

    all_positions = np.stack(all_positions, axis=1)
    return all_positions

#function from paper
def project_points(xy):
    """
    :param xy: [batch_size, horizon, 2]
    :return: [batch_size, horizon, 2]
    """
    batch_size, horizon, _ = xy.shape
    npzfile = np.load("calib_results3.npz")
    ret, mtx, dist, rvecs, tvecs = npzfile["ret"], npzfile["mtx"], npzfile["dist"], npzfile["rvecs"], npzfile["tvecs"]

    # camera is ~0.48m above ground
    xyz = np.concatenate([xy, -0.48 * np.ones(list(xy.shape[:-1]) + [1])], axis=-1) # 0.48
    rvec = tvec = (0, 0, 0)
    camera_matrix = mtx

    # x = y
    # y = -z
    # z = x
    xyz[..., 0] += 0.2 #0.15  # NOTE(greg): shift to be in front of image plane #0.7 seemed good
    xyz_cv = np.stack([xyz[..., 1], -xyz[..., 2], xyz[..., 0]], axis=-1)
    uv, _ = cv2.projectPoints(xyz_cv.reshape(batch_size * horizon, 3), rvec, tvec, camera_matrix, dist)
    uv = uv.reshape(batch_size, horizon, 2)

    return uv

#testing with a fake/generated path
#linvel = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
#ang = np.array([0.2, 0.2, 0.2, 0.3, 0.3, 0.2, 0.1, 0.1])
#img = cv2.imread("data/processed/imgs/frame000878.png")

#testing with an image and corresponding path and imu from real data
img = cv2.imread("data/processed/0405and1605and0106and1506new/imgs/frame001519.png")
csv_df = pd.read_csv("data/processed/0405and1605and0106and1506new/data.csv", header=0)
linvel = np.array(csv_df.iloc[1519, 1:9])  #1519 was a pretty good example
ang = np.array(csv_df.iloc[1519, 9:17])    #1573 turny turn example

r = x/np.tan(-ang) #turning radiuses for all the angles
angvel = linvel/r #angular velocities

pos = commands_to_positions(linvel, angvel)
pixels = project_points(pos)
print("Pixel coordinates:") 
print(pixels)

img_draw = img.copy()

for i in range(len(pixels[0])-1):
    if (pixels[0][i][0] < 0) or (pixels[0][i+1][0] < 0) or (pixels[0][i][1] <0) or (pixels[0][i+1][1]) < 0:
        print("Negative pixel values detected, skipping", i)
        continue
    elif (pixels[0][i][0] > 1490) or (pixels[0][i+1][0] > 1490) or (pixels[0][i][1]>732) or (pixels[0][i+1][1] > 732):
        print("Pixel values out of image. Could not show segment", i)
        continue
    else:
        cv2.line(img_draw, (int(pixels[0][i][0]), int(pixels[0][i][1])), (int(pixels[0][i+1][0]), int(pixels[0][i+1][1])), (255, 0, 0), 3)
        print("Drawing", i)

cv2.imshow("Show path", img_draw)
cv2.waitKey()
cv2.destroyAllWindows()
