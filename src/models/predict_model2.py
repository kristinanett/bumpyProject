import torch
from src.models.model2 import Net
import hydra
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from src.models.bumpy_dataset2 import BumpyDataset
from src.models.bumpy_dataset2 import Rescale, NormalizeIMG, ToTensor, Crop
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
import cv2
import random
log = logging.getLogger(__name__)

def predict(cfg, path):
    train_params = cfg.train.hyperparams
    model_params = cfg.model.hyperparams
    """Runs a prediction script to predict the bumpiness of a path. Uses an image from (..data/example_images.npy)
    and a specific model specified on the command line by eg +config_path=outputs/2022-05-18/22-05-01/.hydra/config.yaml
    """
    log.info("Starting prediction")

    #recreating the model
    model_weights_path = "/".join(path.split("/")[:3])
    state_dict = torch.load(get_original_cwd() + "/" + model_weights_path + "/models/thismodel.pt")
    model = Net(model_params)
    if torch.cuda.is_available():
          log.info('##Converting network to cuda-enabled##')
          model.cuda()
    model.load_state_dict(state_dict)

    #initialize the testset and dataloader
    #the paths will break if model was trained on cluster and now tested on local
    trained_on_cluster = True
    if trained_on_cluster == False:
        dataset = BumpyDataset(
            get_original_cwd() + "/" + train_params.csv_data_path, 
            get_original_cwd() + "/" + train_params.img_data_path, 
            transform=transforms.Compose([Rescale(train_params.img_rescale), Crop(train_params.crop_ratio), NormalizeIMG(), ToTensor()])
            )
    else:
        csv = "data/processed/" + "/".join(train_params.csv_data_path.split("/")[-2:])
        print(csv)
        imgs = "data/processed/" + "/".join(train_params.img_data_path.split("/")[3:5]) + "/"
        dataset = BumpyDataset(
            get_original_cwd() + "/" + csv,
            get_original_cwd() + "/" + imgs, 
            transform=transforms.Compose([Rescale(train_params.img_rescale), Crop(train_params.crop_ratio), NormalizeIMG(), ToTensor()])
            )

    train_size = int(0.8 * len(dataset)) 
    val_size = int(0.15*len(dataset)) 
    test_size = len(dataset) - train_size - val_size 
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator = torch.manual_seed(cfg.train.hyperparams.seed))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=train_params.shuffle, num_workers=4)
    test_loader_iter = iter(test_loader)

    #getting some data
    for i in range(4):
        inputs, labels, idx = next(test_loader_iter)
        #12 asphalt

    if torch.cuda.is_available():
        inputs, labels = [inputs[0].cuda(), inputs[1].cuda(), inputs[2].cuda()] , labels.cuda()

    #creating two routes (left and right) [1, 8, 2])
    fake_lin_coms_left = []
    fake_lin_coms_right = []
    boundaries = [0.45, 0.55]
    for i in range(8):
        linear_x_left = random.uniform(boundaries[0], boundaries[1])
        linear_x_right = random.uniform(boundaries[0], boundaries[1])
        fake_lin_coms_left.append(linear_x_left)
        fake_lin_coms_right.append(linear_x_right)

    path_left = torch.tensor([[[fake_lin_coms_left[0], 0.2], [fake_lin_coms_left[1], 0.2], [fake_lin_coms_left[2], 0.2], [fake_lin_coms_left[3], 0.3], 
                            [fake_lin_coms_left[4], 0.2], [fake_lin_coms_left[5], 0.1], [fake_lin_coms_left[6], 0.2], [fake_lin_coms_left[7], 0.2]]]).cuda()

    path_right = torch.tensor([[[fake_lin_coms_right [0], -0.2], [fake_lin_coms_right [1], -0.2], [fake_lin_coms_right [2], -0.2], [fake_lin_coms_right [3], -0.3], 
                            [fake_lin_coms_right [4], -0.2], [fake_lin_coms_right [5], -0.1], [fake_lin_coms_right [6], -0.2], [fake_lin_coms_right [7], -0.2]]]).cuda()

    #making fake predictions
    model.eval()
    output_left = model([inputs[0], path_left, inputs[2]])
    prediction_left = np.round(output_left[0].cpu().detach().numpy(), 4)

    model.eval()
    output_right = model([inputs[0], path_right, inputs[2]])
    prediction_right = np.round(output_right[0].cpu().detach().numpy(), 4)
    
    #path_comparison_table = np.concatenate((prediction_left, prediction_right), axis=1)
    log.info(f"Predicted normalized imu values for left and right path are: {prediction_left[0], prediction_right[0]}")

    #making a prediction for the actual path and printing it
    model.eval()
    output = model(inputs)
    predicted = np.round(output[0].cpu().detach().numpy(), 4)
    actual = np.round(labels[0].cpu().detach().numpy(), 4)
    actual_mean = np.mean(actual, axis = 0)
    log.info(f"Actual normalized imu values for this path are: \n {actual}")
    log.info(f"Predicted mean normalized imu value for this path is: {predicted}, actual mean is {actual_mean}")

    #showing the denormalized input coms data
    lin_com_mean, lin_com_std, ang_com_mean, ang_com_std, imu_mean, imu_std = dataset.stats
    lin_coms = np.round(((inputs[1][0].cpu()[:, 0].numpy())*lin_com_std)+lin_com_mean, 3)
    ang_coms = np.round(((inputs[1][0].cpu()[:, 1].numpy())*ang_com_std)+(-ang_com_mean), 3)
    log.info(f"The path lin coms are: {lin_coms}")
    log.info(f"The path ang coms are: {ang_coms}")

    #getting the original image to show (the one from dataloader is normalized)
    idx = int(idx.numpy()[0])
    img_name = "frame%06i.png" % idx
    #cv_img = cv2.imread(get_original_cwd() + "/" + "data/processed/imgs/" + img_name)
    cv_img = cv2.imread(get_original_cwd() + "/data/processed/" + "/".join(train_params.img_data_path.split("/")[3:5]) + "/" + img_name)
    rgb_img_original = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB) #(732, 1490, 3)
    h, w = rgb_img_original.shape[:2]
    if h > w:
        new_h, new_w = train_params.img_rescale* h / w, train_params.img_rescale
    else:
        new_h, new_w = train_params.img_rescale, train_params.img_rescale * w / h
    new_h, new_w = int(new_h), int(new_w)
    rescale_img = cv2.resize(rgb_img_original, (new_w, new_h))
    crop_img = rescale_img[int(train_params.crop_ratio*new_h):new_h, :].copy()

    plt.imshow(crop_img)
    os.makedirs("reports/figures/", exist_ok=True)
    plt.savefig("reports/figures/predicted_img.png")
    plt.show()


@hydra.main()
def main(cfg): 
    #Command line call: python src/models/predict_model.py +config_path=outputs/2022-06-21/14-51-34/.hydra/config.yaml #best predicting 1
    #python src/models/predict_model.py +config_path=outputs/2022-06-03/20-01-18/.hydra/config.yaml #best predicting 8
    path = cfg.config_path
    cfg = OmegaConf.load(get_original_cwd() + "/" + path)
    torch.manual_seed(cfg.train.hyperparams.seed)
    predict(cfg, path)

if __name__ == "__main__":
    main()


