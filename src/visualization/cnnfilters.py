import numpy as np 
import hydra
from hydra.utils import get_original_cwd
import omegaconf
import logging
import torch
import torch.nn as nn
log = logging.getLogger(__name__)
from src.models.model3 import Net
import matplotlib.pyplot as plt


def plot_filters_multi_channel(t):
    
    #get the number of kernals
    num_kernels = t.shape[0]    
    
    #define number of columns for subplots
    num_cols = 8
    #rows = num of kernels
    num_rows = 4 # num_kernels
    
    #set the figure size
    fig = plt.figure(figsize=(15, 15))
    
    #looping through all the kernels
    for i in range(t.shape[0]):
        ax1 = fig.add_subplot(num_rows,num_cols,i+1)
        
        #for each kernel, we convert the tensor to numpy 
        npimg = np.array(t[i].numpy(), np.float32)
        #standardize the numpy image
        npimg = (npimg - np.mean(npimg)) / np.std(npimg)
        npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
        npimg = npimg.transpose((1, 2, 0))
        ax1.imshow(npimg)
        ax1.axis('off')
        ax1.set_title(str(i))
        ax1.set_xticklabels([])
        ax1.set_yticklabels([])
        
    plt.savefig('myimage.png', dpi=100)    
    plt.tight_layout()
    plt.show()

def plot_weights(model):
      
  #extracting the model features at the particular layer number
  layer = model.conv_1
  
  #checking whether the layer is convolution layer or not 
  if isinstance(layer, nn.Conv2d):
    #getting the weight tensor data
    weight_tensor = layer.weight.data.cpu()
    
    if weight_tensor.shape[1] == 3:
        plot_filters_multi_channel(weight_tensor)
    else:
        print("Can only plot weights with three channels with single channel = False")
        
  else:
    print("Can only visualize layers which are convolutional")
        


exp_dir = "/outputs/2022-07-13/14-26-00/" #lowpass
csv_path = "data/processed/lowpass8/data.csv" #lowpass
imgs_path = "data/processed/lowpass8/imgs/" #lowpass

@hydra.main()
def main(cfg):
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

    print(model)

    #visualize weights for model - first conv layer
    plot_weights(model)


if __name__ == "__main__":
     main()



