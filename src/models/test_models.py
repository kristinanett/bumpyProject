from importlib.resources import path
import numpy as np 
import hydra
from hydra.utils import get_original_cwd
import omegaconf
import logging
import cv2
import torch
import random
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
from colour import Color
log = logging.getLogger(__name__)

#objective: Feed in an image from the testloader and some paths - run through the model and visualize the predictions as colored paths
#works with 2 fake paths - left and right and also the actual path taken (can be used for predicting or ground truth)

#changable parameters
type = 'lowpass' #type of model that will be used either: regular, lowpass or onevalue
debug_dt = 0.7 #0.25 in the paper #0.6 seems pretty good for me
x = 0.38 #distance from front wheel to robot centre

if type == "onevalue":
    exp_dir = "/outputs/2022-06-21/14-51-34/" #one value
    csv_path = "data/processed/0405and1605and1506new/data.csv" #one value
    imgs_path = "data/processed/0405and1605and1506new/imgs/" #one value
    from src.models.model2 import Net
    from src.models.bumpy_dataset2 import BumpyDataset
    from src.models.bumpy_dataset2 import Rescale, NormalizeIMG, ToTensor, Crop
elif type == 'regular':
    exp_dir = "/outputs/2022-06-13/00-27-50/" #best regular
    csv_path = "data/processed//0405and1605/data.csv" #best regular
    imgs_path = "data/processed//0405and1605/imgs/" #best regular
    from src.models.model import Net
    from src.models.bumpy_dataset import BumpyDataset
    from src.models.bumpy_dataset import Rescale, NormalizeIMG, ToTensor, Crop
else:
    exp_dir = "/outputs/2022-07-13/14-26-00/" #lowpass
    csv_path = "data/processed/lowpass8/data.csv" #lowpass
    imgs_path = "data/processed/lowpass8/imgs/" #lowpass
    from src.models.bumpy_dataset2 import BumpyDataset
    from src.models.bumpy_dataset2 import Rescale, NormalizeIMG, ToTensor, Crop
    from src.models.model3 import Net


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

def project_points(xy):
    """
    :param xy: [batch_size, horizon, 2]
    :return: [batch_size, horizon, 2]
    """
    batch_size, horizon, _ = xy.shape
    npzfile = np.load(get_original_cwd() + "/data/processed/additional/calib_results3.npz")
    ret, mtx, dist, rvecs, tvecs = npzfile["ret"], npzfile["mtx"], npzfile["dist"], npzfile["rvecs"], npzfile["tvecs"]

    # camera is ~0.48m above ground
    xyz = np.concatenate([xy, -0.48 * np.ones(list(xy.shape[:-1]) + [1])], axis=-1) # 0.48
    rvec = tvec = (0, 0, 0)
    camera_matrix = mtx

    # x = y
    # y = -z
    # z = x
    xyz[..., 0] += 0.3 #0.15  # shift to be in front of image plane #0.7 seemed good
    xyz_cv = np.stack([xyz[..., 1], -xyz[..., 2], xyz[..., 0]], axis=-1)
    uv, _ = cv2.projectPoints(xyz_cv.reshape(batch_size * horizon, 3), rvec, tvec, camera_matrix, dist)
    uv = uv.reshape(batch_size, horizon, 2)

    return uv

def generatePaths():
    #creating two routes (left and right) [1, 8, 2])
    fake_lin_coms_left = []
    fake_lin_coms_right = []
    boundaries = [0.45, 0.55]
    for i in range(8):
        linear_x_left = random.uniform(boundaries[0], boundaries[1])
        linear_x_right = random.uniform(boundaries[0], boundaries[1])
        fake_lin_coms_left.append(linear_x_left)
        fake_lin_coms_right.append(linear_x_right)


    fake_ang_coms_left = [0.3,0.3,0.2,0.1,0.0,-0.1,0.0,-0.1]
    fake_ang_coms_right = [-0.3,-0.3,-0.2,-0.1,0.0,0.1,0.0,0.1] # initial path [0.0,-0.2,-0.2,-0.3,-0.2,-0.1,-0.2,-0.2]
    path_left = []
    path_right = []
    for e in range(len(fake_lin_coms_left)):
        path_left.append([fake_lin_coms_left[e], fake_ang_coms_left[e]])
        path_right.append([fake_lin_coms_right[e], fake_ang_coms_right[e]])

    path_left = torch.tensor([path_left]).cuda()
    path_right = torch.tensor([path_right]).cuda()
    
    return path_left, path_right

def drawPath(img_draw, path, prediction, choice, groundtruth=False):
    global type
    if groundtruth:
        thickness = 8
        shift_x = 30
        shift_y = 15
    else:
        thickness = 8
        shift_x = 0
        shift_y = 0
    linvel = path[0][:, 0].cpu().numpy()
    ang = path[0][:, 1].cpu().numpy()

    angvel = (linvel*np.tan(-ang))/x #angular velocities (dividing by the turn radius r)

    #calculating positions and projecting to image plane pixel coordinates
    pos = commands_to_positions(linvel, angvel)
    pixels = project_points(pos)

    #colors
    green = Color("#57bb8a") #Color("green") 
    yellow = Color("#ffd666") #Color("yellow")
    red = Color("#e67c73")#Color("red")
    colors1 = list(green.range_to(yellow,5)) #10 before
    colors2 = list(yellow.range_to(red,5)) #10 before
    colors = colors1 + colors2

    #loop over commands
    for i in range(len(pixels[0])-1):
        if type == "onevalue" and not groundtruth:
            closest_idx = min(range(len(colors)), key=lambda x:abs(x-round(prediction[0]))) #for one prediction per path
        #print("Closest idx:", closest_idx)
        else:
            closest_idx = min(range(len(colors)), key=lambda x:abs(x-round(prediction[0][i]))) #for many predictions
        col = colors[closest_idx]
        col_rgb = tuple([int(255*x) for x in col.rgb])
        if (pixels[0][i][0] < 0) or (pixels[0][i+1][0] < 0) or (pixels[0][i][1] <0) or (pixels[0][i+1][1]) < 0:
            print("Negative pixel values detected, skipping", i)
            continue
        elif (pixels[0][i][0] > 1490) or (pixels[0][i+1][0] > 1490) or (pixels[0][i][1]>732) or (pixels[0][i+1][1] > 732):
            print("Pixel values out of image. Could not show segment", i)
            continue
        elif abs(int(pixels[0][i][0]) - int(pixels[0][i+1][0])) > 300 or abs(int(pixels[0][i][1]) - int(pixels[0][i+1][1])) > 100: #(250 and 70)
            print("Segment start and end are too far apart, skipping", i)
            continue
        else:
            cv2.line(img_draw, (int(pixels[0][i][0]+shift_x), int(pixels[0][i][1]+shift_y)), (int(pixels[0][i+1][0]+shift_x), int(pixels[0][i+1][1]+shift_y)), (col_rgb[2], col_rgb[1], col_rgb[0]), thickness)
            cv2.putText(img_draw, str(choice), (10,50), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 3, 2)
            print("Drawing", i)

    return img_draw

def denormalizeIMU(csv_path, predicted_imu):
    
    csv_df = pd.read_csv(get_original_cwd() + "/" + csv_path, header=0)

    if type == "regular":
        imu_all = np.array([csv_df.iloc[:, 16:]])
    else:
        imu_all = np.array([csv_df.iloc[:, 17:]])

    imu_mean, imu_std = np.mean(imu_all), np.std(imu_all)
    denormalized_imu = (predicted_imu*imu_std)+imu_mean

    return denormalized_imu

def denormalizeCOMS(csv_path, coms):
    csv_df = pd.read_csv(get_original_cwd() + "/" + csv_path, header=0)

    if type == "regular":
        lin_com_all = np.array([csv_df.iloc[:, :8]])
        ang_com_all = np.array([csv_df.iloc[:, 8:16]])
    else:
        lin_com_all = np.array([csv_df.iloc[:, 1:9]])
        ang_com_all = np.array([csv_df.iloc[:, 9:17]])

    lin_com_mean, lin_com_std = np.mean(lin_com_all), np.std(lin_com_all)
    ang_com_mean, ang_com_std = np.mean(ang_com_all), np.std(ang_com_all)

    denormalized_lin_coms = (coms[0][:, 0].cpu().numpy()*lin_com_std)+lin_com_mean
    denormalized_ang_coms = (coms[0][:, 1].cpu().numpy()*ang_com_std)+ang_com_mean

    return torch.from_numpy(np.array([np.concatenate(([denormalized_lin_coms], [denormalized_ang_coms]), axis = 0).T]))

@hydra.main()
def main(cfg):
    global type
    global exp_dir

    exp_dir = get_original_cwd() + exp_dir 
    
    #define experiment folder (to get model weights)
    hydra_path = exp_dir + ".hydra/config.yaml"

    #load the config
    cfg = omegaconf.OmegaConf.load(hydra_path)
    train_params = cfg.train.hyperparams
    model_params = cfg.model.hyperparams
    torch.manual_seed(cfg.train.hyperparams.seed)

    #recreating the model
    state_dict = torch.load(exp_dir + "models/thismodel.pt")
    model = Net(model_params)
    if torch.cuda.is_available():
          print('##Converting network to cuda-enabled##')
          model.cuda()
    model.load_state_dict(state_dict)

    #initialize the testset and dataloader
    #the paths are hardcoded in the beginning because they will break if model was trained on cluster and now tested on local

    dataset = BumpyDataset(
        get_original_cwd() + "/" + csv_path,
        get_original_cwd() + "/" + imgs_path, 
        transform=transforms.Compose([Rescale(train_params.img_rescale), Crop(train_params.crop_ratio), NormalizeIMG(), ToTensor()])
        )

    train_size = int(0.8 * len(dataset)) 
    val_size = int(0.15*len(dataset)) 
    test_size = len(dataset) - train_size - val_size 
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator = torch.manual_seed(cfg.train.hyperparams.seed))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=train_params.shuffle, num_workers=4)

    #getting some data from the testloader
    for i, data in enumerate(test_loader, 0):

        #data is in the format: [sample['image'], coms_final, sample['cur_imu']], imu_final, idx
        inputs, labels, idx = data

        if torch.cuda.is_available():
            if type == "regular":
                inputs, labels = [inputs[0].cuda(), inputs[1].cuda()] , labels.cuda()
            else:
                inputs, labels = [inputs[0].cuda(), inputs[1].cuda(), inputs[2].cuda()] , labels.cuda()

        #generating a left and a right path
        path_left, path_right = generatePaths()

        #making predictions for the fake paths
        model.eval()
        if type == "regular":
            output_left = model([inputs[0], path_left])
            output_right = model([inputs[0], path_right])
        else:
            output_left = model([inputs[0], path_left, inputs[2]])
            output_right = model([inputs[0], path_right, inputs[2]])

        prediction_left = np.round(output_left[0].cpu().detach().numpy(), 4)
        prediction_right = np.round(output_right[0].cpu().detach().numpy(), 4)

        #making a prediction for the actual path
        output = model(inputs)
        predicted = np.round(output[0].cpu().detach().numpy(), 4)
        actual = np.round(labels[0].cpu().detach().numpy(), 4)

        #denormalize the predicted imu values
        prediction_left = denormalizeIMU(csv_path, prediction_left)
        prediction_right = denormalizeIMU(csv_path, prediction_right)
        predicted = denormalizeIMU(csv_path, predicted)
        actual = denormalizeIMU(csv_path, actual)

        print(np.mean(prediction_left[:4]))
        print(np.mean(prediction_right[:4]))
        #choose the path with the lower sum of 8 denormalized imu values
        if np.mean(prediction_left[:4]) > np.mean(prediction_right[:4]):
            choice = "Right"
        elif np.mean(prediction_left[:4]) < np.mean(prediction_right[:4]):
            choice = "Left"
        else:
            choice = "Unclear"

        #getting the original image for drawing (the one from dataloader is normalized)
        idx = int(idx.numpy()[0])
        img_name = "frame%06i.png" % idx
        cv_img = cv2.imread(get_original_cwd() + "/" + imgs_path + img_name)
        img_draw = cv_img.copy()

        img_draw = drawPath(img_draw, path_left, prediction_left.T, choice)
        img_draw = drawPath(img_draw, path_right, prediction_right.T, choice)
        #img_draw = drawPath(img_draw, denormalizeCOMS(csv_path, inputs[1]), predicted.T, choice, groundtruth = False)
        #img_draw = drawPath(img_draw, denormalizeCOMS(csv_path, inputs[1]), actual.T, choice, groundtruth = True) 
        # change in last line if need to have onevalue groundtruth as mean np.array([np.mean(actual.T)])
 
        cv2.imshow("Show path", img_draw)

        key = cv2.waitKey(0)
        while key not in [ord('q'), ord('k')]:
            key = cv2.waitKey(0)
        if key == ord('q'):
            break

cv2.destroyAllWindows()

if __name__ == "__main__":
     main()


