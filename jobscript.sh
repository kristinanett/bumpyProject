#!/bin/sh
### General options
### –- specify queue --
#BSUB -q gpuv100
### -- set the job Name --
#BSUB -J train_bumpy1
### -- ask for number of cores (default: 1) --
#BSUB -n 4
#BSUB -R "span[hosts=1]"
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 24:00
# request 5GB of system-memory
#BSUB -R "rusage[mem=5GB]"
### -- set the email address --
# please uncomment the following line and put in your e-mail address,
# if you want to receive e-mail notifications on a non-default address
##BSUB -u kristinanett@gmail.com
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o cluster_output/gpu_%J.out
#BSUB -e cluster_output/gpu_%J.err
# -- end of LSF options --

#loading modules
module load python3/3.9.10
module load cuda/11.3
module load numpy/1.22.2-python-3.9.10-openblas-0.3.19
module load pandas/1.4.1-python-3.9.10
module load matplotlib/3.5.1-numpy-1.22.2-python-3.9.10
export PYTHONPATH=/zhome/d7/e/154401/bumpyProject:$PYTHONPATH
source ../projectenv/bin/activate

python src/models/train_model.py 'train.hyperparams.crop_ratio=0.0' 'train.hyperparams.exp_group="crop experiments"' 'train.hyperparams.comment="0405and1605 no cropping"' 'model.hyperparams.img_height=122'