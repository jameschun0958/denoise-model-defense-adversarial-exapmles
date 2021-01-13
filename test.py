import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split,TensorDataset
import torchvision
from torchvision import transforms, datasets
import torchvision.models as models

from train import Trainer
from models import WideResNet
from models import Model_ComDefend
from utils import progress_bar

import os
import argparse
import utils
import json
import numpy as np

import torchattacks
from PIL import Image


if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    device='cuda'
else:
    device='cpu'

class Normalize:
    """Applies the :class:`~torchvision.transforms.Normalize` transform to a batch of images.
    .. note::
        This transform acts out of place by default, i.e., it does not mutate the input tensor.
    Args:
        mean (sequence): Sequence of means for each channel.
        std (sequence): Sequence of standard deviations for each channel.
        inplace(bool,optional): Bool to make this operation in-place.
        dtype (torch.dtype,optional): The data type of tensors to which the transform will be applied.
        device (torch.device,optional): The device of tensors to which the transform will be applied.
    """

    def __init__(self, mean, std, inplace=False, dtype=torch.float, device='cpu'):
        self.mean = torch.as_tensor(mean, dtype=dtype, device=device)[None, :, None, None]
        self.std = torch.as_tensor(std, dtype=dtype, device=device)[None, :, None, None]
        self.inplace = inplace
        
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor of size (N, C, H, W) to be normalized.
        Returns:
            Tensor: Normalized Tensor.
        """
        if not self.inplace:
            tensor = tensor.clone()

        tensor.sub_(self.mean).div_(self.std)
        return tensor
    
def load_cifar():
    transform = transforms.Compose([transforms.ToTensor()])
            
    #test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    test_dataset = datasets.ImageFolder(root='/train-data/ioc5009/data/adv_cifar10_20', transform=transform)
    print("Image Shape: {}".format(test_dataset[0][0].numpy().shape), end = '\n\n')
    print("Testing Set:       {} samples".format(len(test_dataset)))

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=32)
    
    return test_loader

def load_imagenet():
    transform = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor()
    ])

    # random choose 10000 sample
    dataset = datasets.ImageFolder(root='/train-data/ILSVRC2012/val', transform=transform)
    test_dataset, val_dataset = random_split(dataset, (int(len(dataset) * 0.2), int(len(dataset) * 0.8)))

    print("Image Shape: {}".format(test_dataset[0][0].numpy().shape), end = '\n\n')
    print("Testing Set:       {} samples".format(len(test_dataset)))

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=32)

    return test_loader

def load_adv(adv_data_path):
    adv_images, adv_labels = torch.load(adv_data_path)
    adv_data = TensorDataset(adv_images.float()/255, adv_labels)
    adv_loader = DataLoader(adv_data, batch_size=16, shuffle=False, num_workers=32)

    return adv_loader

def attack(test_model, denoise_model, data_loader, adv_loader, type='cifar10'):
    test_model.to(device).eval()
    denoise_model.to(device).eval()

    if(type=='cifar10'):
        transformer = transforms.Compose([Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), device=device)])
    elif(type=='imagenet'):
        transformer = transforms.Compose([Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225], device=device)])

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for idx, (test_x, test_y) in enumerate(adv_loader):
            test_x, test_y = test_x.to(device), test_y.to(device)
            test_x = transformer(test_x)
            outputs = test_model(test_x)

            _, predicted = outputs.max(1)
            
            total += test_y.size(0)
            correct += (predicted == test_y).sum().item()

            progress_bar(
                idx, len(adv_loader), 
                'Acc: %.3f%% (%d/%d)' %(100. * correct / total, correct, total))

    print('After attack accuracy: %.2f %%' % (100 * float(correct) / total))

    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for idx, (test_x, test_y) in enumerate(adv_loader):
            test_x, test_y = test_x.to(device), test_y.to(device)

            _,_, denoise_out = denoise_model(test_x)
            denoise_out = transformer(denoise_out)

            outputs = test_model(denoise_out)

            _, predicted = outputs.max(1)
            
            total += test_y.size(0)
            correct += (predicted == test_y).sum().item()

            progress_bar(
                idx, len(adv_loader), 
                'Acc: %.3f%% (%d/%d)' %(100. * correct / total, correct, total))

    print('After denoise accuracy: %.2f %%' % (100 * float(correct) / total))

def atks_test(test_model, atks):
    for atk in atks :
        print("-"*70)
        print(atk)
        
        transformer = transforms.Compose([Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), device=device)])
        correct = 0
        total = 0
        
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            adv_images = atk(images, labels)
            adv_images = transformer(adv_images)
            
            outputs = test_model(adv_images)

            _, predicted = outputs.max(1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()
        print('Robust accuracy: %.2f %%' % (100 * float(correct) / total))
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='testing comdefend model')
    parser.add_argument('--test_data', default='cifar10', type=str)
    parser.add_argument('--gen_adv', default=1, type=int)
    args = parser.parse_args()

    denoise_model = Model_ComDefend.comdefend(noise=True, inference=True)

    if args.test_data == 'cifar10':
        print("Laoding cifar10 dataset....")
        test_loader  = load_cifar()

        # Loading pretrained model
        test_model = WideResNet.WideResNet(depth=40, widen_factor=4, num_classes=10)
        checkpoint = torch.load("./models/pretrained_weight/CIAFR10-WRN40-4.pt")
        test_model.load_state_dict(checkpoint['net'])
        test_model.to(device).eval()

        transformer = transforms.Compose([Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), device=device)])
    elif args.test_data == 'imagenet':
        print("Laoding imagenet dataset....")
        test_loader = load_imagenet()
        test_model = models.resnet50(pretrained=True).to(device).eval()

        transformer = transforms.Compose([Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225], device=device)])
    
    # Loading denoise model
    denoise_model.load_state_dict(torch.load("./experiments/CIFAR10-ComDefend/weights/CIFAR10-ComDefend.pt"))

    # Define atk methods
    atks = [
            torchattacks.GN(test_model, sigma=0.1),
            torchattacks.FGSM(test_model, eps=8/255),
            torchattacks.BIM(test_model, eps=8/255),
            torchattacks.PGD(test_model, eps=8/255), # black box atk
            torchattacks.FFGSM(test_model, eps=8/255)
        ]    
    
    #atks_test(test_loader, atks)
    
    # Generating adv sample and save
    if args.gen_adv == 1:
        for atk in atks :
            print("-"*70)
            print(atk)
            atk_name = str(atk).split('(')[0]
            atk.set_return_type('int') # Save as integer.
            save_path = "./data/" + args.test_data + "_" + atk_name + ".pt"
            atk.save(data_loader=test_loader, save_path=save_path, verbose=True)
    
    # Testing model clean accuracy
    print("Testing clean accuracy....")
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for idx, (test_x, test_y) in enumerate(test_loader):
            test_x, test_y = test_x.to(device), test_y.to(device)
            _,_, test_x = denoise_model.to(device).eval()(test_x)

            test_x = transformer(test_x)
            outputs = test_model(test_x) 

            _, predicted = outputs.max(1)
            
            total += test_y.size(0)
            correct += (predicted == test_y).sum().item()

            progress_bar(
                idx, len(test_loader), 
                'Acc: %.3f%% (%d/%d)' %(100. * correct / total, correct, total))
        print('Clean accuracy: %.2f %%' % (100 * float(correct) / total))

    
    # Testing model robust accuracy and after denoise accuracy 
    for atk in atks :
        print("-"*70)
        atk_name = str(atk).split('(')[0]
        adv_path = "./data/" + args.test_data + "_" + atk_name + ".pt"
        print("Loading"+ adv_path + "....")
        adv_loader = load_adv(adv_path)
        attack(test_model, denoise_model, test_loader, adv_loader, type=args.test_data)
    


