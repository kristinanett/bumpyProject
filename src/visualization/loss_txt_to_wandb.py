import numpy as np 
import wandb
import hydra
from hydra.utils import get_original_cwd
import omegaconf
import logging
log = logging.getLogger(__name__)

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

    log.info("Converting run losses to wandb")
    #define txt file with losses and hydra file with config
    exp_dir = get_original_cwd() + "/outputs/2022-06-02/12-39-00/"
    txt_path =  exp_dir + "train_model.log"
    hydra_path = exp_dir + ".hydra/config.yaml"

    #load config
    cfg = omegaconf.OmegaConf.load(hydra_path)
    myconfig = omegaconf.OmegaConf.to_container(cfg, resolve=True, throw_on_missing=True) 

    train_params = cfg.train.hyperparams
    model_params = cfg.model.hyperparams
    wandb.init(config = myconfig, project='bumpyProject', group = "old experiments", notes="0405+1605 data, increased LSTM layers from 1 to 8")
    #wandb.init(project='bumpyProject', group = "old experiments", notes="0405 data first training baseline")

    #get the losses from the txt file
    losses = getLosses(txt_path)


    #for plotting a constant baseline
    # train_tuple = ("train", 0)
    # val_tuple = ("val", 0.98)

    # train_losses = []
    # for i in range(31):
    #     train_losses.append(train_tuple)

    # losses = []
    # for i in range(20):
    #     losses = losses + train_losses
    #     losses.append(val_tuple)

    #fake the training
    for id, loss in losses:
        if id == "train":
            wandb.log({"train loss": loss})
        elif id == "val":
            wandb.log({"validation loss": loss})
       

if __name__ == "__main__":
     main()