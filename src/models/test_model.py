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
import torch.nn as nn
log = logging.getLogger(__name__)

#changable parameters
type = 'onevalue' #type of model that will be used either: regular, lowpass or onevalue

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

    #getting the number of model parameters - got 7841216
    # pytorch_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    # pytorch_total_params = sum(p.numel() for p in model.parameters())
    # params3 = sum(dict((p.data_ptr(), p.numel()) for p in model.parameters()).values())
    # print("Number of total params", pytorch_total_params)
    # print("Number of trainable params", pytorch_train_params)
    # print("Number of trainable params2", params3)

    # initialize the testset and dataloader
    # the paths are hardcoded in the beginning because they will break if model was trained on cluster and now tested on local

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
    print(len(test_set))

    criterion = nn.MSELoss()

    #getting some data from the testloader
    model.eval()
    test_losses = []
    test_loss = 0.0
    for i, data in enumerate(test_loader, 0):

        #data is in the format: [sample['image'], coms_final, sample['cur_imu']], imu_final, idx
        inputs, labels, idx = data
        labels = torch.mean(labels, dim=1) #from [32, 8, 1] to [32, 1] -predicting only mean imu value for each path

        if torch.cuda.is_available():
            if type == "regular":
                inputs, labels = [inputs[0].cuda(), inputs[1].cuda()] , labels.cuda()
            else:
                inputs, labels = [inputs[0].cuda(), inputs[1].cuda(), inputs[2].cuda()] , labels.cuda()

        #making a prediction for the actual path
        output = model(inputs)
        batch_loss = criterion(output, labels)

        test_loss += batch_loss.item()

        if i % 10 == 9:  # print and save average validation loss for every epoch
            log.info("[%d, %5d] validation loss: %.3f" % (0 + 1, i + 1, test_loss / 10))
            test_losses.append(test_loss / 10)  # (loss.data.numpy())
            test_loss = 0.0
        
    log.info("Finished Testing")
    avg = np.sum(np.array(test_losses))/len(test_losses)
    log.info(f"Average loss was, {avg}")

if __name__ == "__main__":
     main()


