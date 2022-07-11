from itertools import count
import numpy as np
import cv2
import pandas as pd
import glob
from colour import Color

debug_dt = 0.7 #0.25 in the paper #0.6 seems pretty good for me
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
    npzfile = np.load("data/processed/additional/calib_results3.npz")
    ret, mtx, dist, rvecs, tvecs = npzfile["ret"], npzfile["mtx"], npzfile["dist"], npzfile["rvecs"], npzfile["tvecs"]

    # camera is ~0.48m above ground
    xyz = np.concatenate([xy, -0.48 * np.ones(list(xy.shape[:-1]) + [1])], axis=-1) # 0.48
    rvec = tvec = (0, 0, 0)
    camera_matrix = mtx

    # x = y
    # y = -z
    # z = x
    xyz[..., 0] += 0.3 #0.15  # NOTE(greg): shift to be in front of image plane #0.7 seemed good
    xyz_cv = np.stack([xyz[..., 1], -xyz[..., 2], xyz[..., 0]], axis=-1)
    uv, _ = cv2.projectPoints(xyz_cv.reshape(batch_size * horizon, 3), rvec, tvec, camera_matrix, dist)
    uv = uv.reshape(batch_size, horizon, 2)

    return uv

#testing with a fake/generated path
# linvel = np.array([0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5])
# ang = np.array([0.2, 0.2, 0.2, 0.3, 0.3, 0.2, 0.1, 0.1])
# img = cv2.imread("data/processed/imgs/frame000878.png")

img_idx = 0 #11100
#2605, 3716, 3769 is an okay example for lowpass vs nopass
#5225, 5444 nice tall grass with nopass (lopass and nopass work the same)
#2488, 2538 is good with lowpass
#3762 good with nopass
#11100 nopass has no difference between asphalt and low grass 
#7400 same vid as in the presentation
#5376 tall grass with some smoother 
#176 small grass is apparently better than asphalt

while True:
    dir = "data/processed/lowpass10/"
    #dir = "data/processed/lowpass11/"
    #dir = "data/processed/0405and1605and0106and1506new/"
    #testing with an image and corresponding path and imu from real data
    img = cv2.imread(dir+"imgs/" + "frame%06i.png" % img_idx)
    csv_df = pd.read_csv(dir +"data.csv", header=0)
    linvel = np.array(csv_df.iloc[img_idx, 1:9])  #1519 was a pretty good example #1553 from asphalt to grass
    ang = np.array(csv_df.iloc[img_idx, 9:17])    #1573 turny turn example
    imu = np.array([csv_df.iloc[img_idx, 17:]])

    angvel = (linvel*np.tan(-ang))/x #angular velocities (dividing by the turn radius r)

    #calculating positions and projecting to image plane pixel coordinates
    pos = commands_to_positions(linvel, angvel)
    #print(pos)
    #print(pos.shape) #(8, 9, 2)
    pixels = project_points(pos)
    #print("Pixel coordinates:")
    #print(pixels)
    #print(pixels.shape) #(8, 9, 2)

    img_draw = img.copy()

    #colors
    green = Color("#57bb8a") #Color("green") 
    yellow = Color("#ffd666") #Color("yellow")
    red = Color("#e67c73")#Color("red")
    colors1 = list(green.range_to(yellow,5)) #10 before
    colors2 = list(yellow.range_to(red,5)) #10 before
    colors = colors1 + colors2
    #print("IMU values", imu)

    #loop over commands
    for i in range(len(pixels[0])-1):
        closest_idx = min(range(len(colors)), key=lambda x:abs(x-round(imu[0][i])))
        #print("Closest idx:", closest_idx)
        col = colors[closest_idx]
        col_rgb = tuple([int(255*x) for x in col.rgb])
        if (pixels[0][i][0] < 0) or (pixels[0][i+1][0] < 0) or (pixels[0][i][1] <0) or (pixels[0][i+1][1]) < 0:
            print("Negative pixel values detected, skipping", i)
            continue
        elif (pixels[0][i][0] > 1490) or (pixels[0][i+1][0] > 1490) or (pixels[0][i][1]>732) or (pixels[0][i+1][1] > 732):
            print("Pixel values out of image. Could not show segment", i)
            continue
        elif abs(int(pixels[0][i][0]) - int(pixels[0][i+1][0])) > 250 or abs(int(pixels[0][i][1]) - int(pixels[0][i+1][1])) > 70:
            print("Segment start and end are too far apart, skipping", i)
            continue
        else:
            cv2.line(img_draw, (int(pixels[0][i][0]), int(pixels[0][i][1])), (int(pixels[0][i+1][0]), int(pixels[0][i+1][1])), (col_rgb[2], col_rgb[1], col_rgb[0]), 3)
            cv2.putText(img_draw, str(img_idx), (10,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 1, 2)
            print("Drawing", i)

    cv2.imshow("Show path", img_draw)
    img_idx += 1

    key = cv2.waitKey(0)
    while key not in [ord('q'), ord('k')]:
        key = cv2.waitKey(0)
    if key == ord('q'):
        break

cv2.destroyAllWindows()
