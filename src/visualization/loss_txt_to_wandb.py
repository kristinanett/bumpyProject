import numpy as np 
import wandb
import hydra
from hydra.utils import get_original_cwd
import omegaconf

#reading training and validation losses from file
def getLosses(txt_path):
    f = open(txt_path, "r")
    lines = f.readlines()

    losses = []
    val_losses_small = []

    for line in lines:
        if "train loss" in line:
            if val_losses_small:
                losses.append(("val", np.mean(val_losses_small)))
                val_losses_small = []
            losses.append(("train", float(line.split(" ")[-1].strip())))
        elif "validation loss" in line:
            val_losses_small.append(float(line.split(" ")[-1].strip()))
    
    if val_losses_small:
        losses.append(("val", np.mean(val_losses_small)))

    f.close()
    return losses



@hydra.main()
def main(cfg):

    #define txt file with losses and hydra file with config
    exp_dir = get_original_cwd() + "/outputs/2022-05-18/22-05-01/" #"/outputs/2022-06-03/19-01-30/"
    txt_path =  exp_dir + "train_model.log"
    hydra_path = exp_dir + ".hydra/config.yaml"

    #load config
    cfg = omegaconf.OmegaConf.load(hydra_path)
    myconfig = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True) 

    train_params = cfg.train.hyperparams
    model_params = cfg.model.hyperparams
    wandb.init(config = myconfig, project='bumpyProject', group = "old experiments", notes="0405 data first training run 20 epochs")

    #get the losses from the txt file
    losses = getLosses(txt_path)

    #fake the training
    for id, loss in losses:
        if id == "train":
            wandb.log({"train loss": loss})
        elif id == "val":
            wandb.log({"validation loss": loss})
       

if __name__ == "__main__":
     main()