# MLFSR
Mamba-based Light Field Super-Resolution with Efficient Subspace Scanning (ACCV 24)


### 1. environment setup

CUDA >= 11.7 \
causal-conv1d 1.1.3 \
mamba-ssm 1.1.4 \
torch 2.1 \
torchvision 0.16.0 

```pip install h5py einops xlwt numpy==1.21.1 thop timm accelerate```


### 2. Data preparation

Set patch size and stride in ```Generate_Data_for_Training.py``` and ```Generate_Data_for_Test.py``` to generate training and test data.


### 3. Evaluation

Set *test_mode* in ```test_option.py``` and run ```test.py```.

### 4. Training

Set parameters in ```option.py``` and run ```train.py``` or ```train_multi.py```(multiple GPUs).


## Acknowledgement
This repository is based on the [BasicLFSR](https://github.com/ZhengyuLiang24/BasicLFSR).

