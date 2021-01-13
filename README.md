# Comdefend_pytorch

## Requires
- Pytorch 1.7.0
- Python 3.6.9
- torchattacks 2.12.1
- numpy 1.18.5
- matplotlib 3.3.1

## Note
! You also have to prepare pre-trained weight and model to the corresponding folder by yourself.


## Usage
For training denoise model
```
python3 main.py --model_dir experiments/CIFAR10-ComDefend
```

For testing robustness of pretrained model and effectiveness of denoise model 
```
python3 test.py --test_data cifar10 --gen_adv 1
```