# Signal Temporal Logic Neural Predictive Control
> (non-official impl.) A differentiable learning framework to define task requirements and to learn control policies for robots.

## Prerequisite
Ubuntu 20.04+ (better to have a GPU like NVidia RTX 2080Ti)

Packages:
1. Numpy and Matplotlib: `conda install numpy matplotlib`
2. PyTorch v1.13.1 [[link]](https://pytorch.org/get-started/previous-versions/): `conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia` (other version might also work )

## Command
`python train_nn.py -e e1_traffic --lr 5e-4`

results will be saved in `./exps_stl/gxxxx-yyyyyy_e1_traffic/log-xxxx-yyyyyy.txt`
