import torch
from src.models.model import Net
import hydra
import argparse
from omegaconf import OmegaConf

def predict(cfg, path):

    """Function for making predictions given a model and input data"""
    model_params = cfg.model.hyperparams
    model_dict_path = path.split("/")[:3]

    state_dict = torch.load(model_dict_path + "/models/thismodel.pt")
    model = Net(model_params)
    model.load_state_dict(state_dict)


@hydra.main()
def main(cfg):
    #Command line call: python infer.py +config_path=outputs/2022-05-18/22-05-01/.hydra/config.yaml
    path = cfg.config_path
    cfg = OmegaConf.load(path)
    torch.manual_seed(cfg.train.hyperparams.seed)
    predict(cfg, path)

if __name__ == "__main__":
    main()