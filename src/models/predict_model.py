import torch
from src.models.model import Net
import hydra
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf
from src.models.bumpy_dataset import BumpyDataset
from src.models.bumpy_dataset import Rescale, Normalize, ToTensor
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import numpy as np
import matplotlib.pyplot as plt
import logging
import os
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
    dataset = BumpyDataset(
        get_original_cwd() + "/" + cfg.train.hyperparams.csv_data_path, 
        get_original_cwd() + "/" + train_params.img_data_path, 
        transform=transforms.Compose([Rescale(train_params.img_rescale), Normalize(), ToTensor()])
        )

    train_size = int(0.8 * len(dataset)) #18624
    val_size = int(0.15*len(dataset)) #3492
    test_size = len(dataset) - train_size - val_size #1164
    train_set, val_set, test_set = torch.utils.data.random_split(dataset, [train_size, val_size, test_size], generator = torch.manual_seed(cfg.train.hyperparams.seed))
    test_loader = DataLoader(test_set, batch_size=1, shuffle=train_params.shuffle, num_workers=4)
    test_loader_iter = iter(test_loader)

    #getting some data
    for i in range(3):
        inputs, labels = next(test_loader_iter)

    if torch.cuda.is_available():
        inputs, labels = [inputs[0].cuda(), inputs[1].cuda()] , labels.cuda()
    
    #making a prediction and printing it
    #Stats: 0.49952415462324845 0.0288900208592242 -0.010268470790378011 0.24212891170241857 9.19860582807647 13.002102541602538
    model.eval()
    output = model(inputs)
    predicted = np.round(output[0].cpu().detach().numpy(), 4)
    actual = np.round(labels[0].cpu().detach().numpy(), 4)
    comparison_table = np.concatenate((actual, predicted), axis=1)
    log.info(f"Actual (column0) VS predicted (column1) normalized imu values for this path are: \n {comparison_table}")

    #showing the denormalized input data
    lin_coms = np.round(((inputs[1][0].cpu()[:, 0].numpy())*0.0288900208592242)+0.49952415462324845, 3)
    ang_coms = np.round(((inputs[1][0].cpu()[:, 1].numpy())*0.24212891170241857)+(-0.010268470790378011), 3)
    log.info(f"The path lin coms are: {lin_coms}")
    log.info(f"The path ang coms are: {ang_coms}")
    img_to_show = (np.moveaxis(inputs[0][0].cpu().numpy(), 0, -1))  #(122, 248, 3)
    img_denormalized = ((img_to_show*127.5)+127.5).astype(int) #before floats -1 to 1, after ints 0 to 255

    plt.imshow(img_denormalized)
    os.makedirs("reports/figures/", exist_ok=True)
    plt.savefig("reports/figures/predicted_img.png")
    plt.show()

@hydra.main()
def main(cfg):
    #Command line call: python src/models/predict_model.py +config_path=outputs/2022-05-26/14-07-07/.hydra/config.yaml
    path = cfg.config_path
    cfg = OmegaConf.load(get_original_cwd() + "/" + path)
    torch.manual_seed(cfg.train.hyperparams.seed)
    predict(cfg, path)

if __name__ == "__main__":
    main()


