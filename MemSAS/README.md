## Introduction
Pytorch implementation of paper: MemSAS: Prototype-driven Spatio-Temporal Memory with Task-Adaptive Decouple Features for Skeleton-Based Action Segmentation.

The code is modified from [ME-ST](https://github.com/HaoyuJi/ME-ST)
## Enviroment
Pytorch == `1.10.1+cu111`, 
torchvision == `0.11.2`, 
python == `3.8.13`, 
CUDA==`11.4`

### Enviroment Setup
Within the newly instantiated virtual environment, execute the following command to install all dependencies listed in the `requirements.txt` file.

``` python
pip install -r requirements.txt
```

## Preparation
### Datasets
This study conducted experiments on four publicly available datasets: LARa, TCG, PKU-MMD (X-sub), and PKU-MMD (X-view). 
All datasets are downloadable from 
[ME-ST](https://github.com/HaoyuJi/ME-ST)


## Get Started

### Training

To train our model on different datasets (taking the LARa dataset as an example), use the following commands:

MemSAS without BRB:
```shell
python MemSAS_train.py --dataset lara --cuda 0
```

MemSAS with BRB:
```shell
python MemSAS+BRB_train.py --dataset lara --cuda 0
```

### Evaluation

To evaluate the performance of the results obtained after the training, use the following commands:


ME-ST without BRB:
```shell
python MemSAS_test.py --dataset lara --cuda 0 --model_path pre_trained_models/ME-ST/lara/best_model.pt
```

ME-ST with BRB:
```shell
python MemSAS+BRB_test.py --dataset lara --cuda 0 --model_path pre_trained_models/ME-ST+BRB/lara/best_model.pt
```
## Acknowledgement
The MemSAS model and code are built upon [ME-ST](https://github.com/HaoyuJi/ME-ST)
