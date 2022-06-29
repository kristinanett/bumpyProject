import numpy as np
import cv2
debug_dt = 0.25 #random number from the paper
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
    print(rvecs)
    print(tvecs)

    # camera is ~0.48m above ground
    xyz = np.concatenate([xy, -0.48 * np.ones(list(xy.shape[:-1]) + [1])], axis=-1) # 0.48
    rvec = tvec = (0, 0, 0)
    camera_matrix = mtx

    # x = y
    # y = -z
    # z = x
    xyz[..., 0] += 0.15  # NOTE(greg): shift to be in front of image plane
    xyz_cv = np.stack([xyz[..., 1], -xyz[..., 2], xyz[..., 0]], axis=-1)
    uv, _ = cv2.projectPoints(xyz_cv.reshape(batch_size * horizon, 3), rvec, tvec, camera_matrix, dist)
    uv = uv.reshape(batch_size, horizon, 2)

    return uv


linvel = np.array([[0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5]])

ang = np.array([[0.2, 0.2, 0.2, 0.3, 0.3, 0.2, 0.1, 0.1]])
r = x/np.tan(-ang) #turning radiuses for all the angles
angvel = linvel/r #angular velocities

pos = commands_to_positions(linvel, angvel)
pixels = project_points(pos)
print(pos)
print(pixels)

x_list, y_list = [], []
img = cv2.imread("data/processed/imgs/frame000878.png")
im_lims = img.shape
for pix in pixels:
    pix_lims = (732., 1490.) #(480., 640.)

    assert pix_lims[1] / pix_lims[0] == im_lims[1] / float(im_lims[0])
    resize = im_lims[0] / pix_lims[0]

    pix = resize * pix
    x_list.append(im_lims[1] - pix[:, 0])
    y_list.append(im_lims[0] - pix[:, 1])

print(x_list) #weird values (not on image)
print(y_list) #weird values (not on image)

img_draw = img.copy()

for i in range(len(x_list)-1):
    cv2.line(img_draw, (x_list[i], y_list[i]), (x_list[i+1], y_list[i+1]), (255, 0, 0), 2)

cv2.imshow("Show path", img_draw)
cv2.waitKey()
cv2.destroyAllWindows()
