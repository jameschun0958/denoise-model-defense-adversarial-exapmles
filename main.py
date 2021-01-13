import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, RandomSampler
from torchvision import transforms, datasets
import torchvision.models as models

from train import Trainer
from models import mobilenet
from models import WideResNet
from models import Model_ComDefend

import os
import argparse
import utils
import json
import numpy as np

parser = argparse.ArgumentParser(description='PyTorch ComDefend training')
parser.add_argument('--model_dir', default='experiments/test/',
                    help="Directory containing params.json")

# Load the parameters from json file
args = parser.parse_args()
json_path = os.path.join(args.model_dir, 'params.json')
assert os.path.isfile(json_path), "No json configuration file found at {}".format(json_path)
params = utils.Params(json_path)

seed=8888

if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.cuda.manual_seed_all(seed)

def load_cifar():
    transform_train = transforms.Compose([transforms.ToTensor()])
    
    transform_test = transforms.Compose([transforms.ToTensor()])
            
    train_dataset = datasets.CIFAR10('./data', train=True, download=True, transform=transform_train)
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform_test)
    
    print("Image Shape: {}".format(train_dataset[0][0].numpy().shape), end = '\n\n')
    print("Training Set:   {} samples".format(len(train_dataset)))
    print("Test Set:       {} samples".format(len(test_dataset)))
    
    BATCH_SIZE = params.batch_size
    NUM_WORKERS = params.num_workers

    # Create iterator.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS)
    
    return train_loader, test_loader

def load_imagenet():
    transform = transforms.Compose([
        transforms.Resize((224,224)), 
        transforms.ToTensor()
    ])

    # random choose 10000 sample
    train_dataset = datasets.ImageFolder(root='/train-data/ILSVRC2012/train', transform=transform)
    test_dataset = datasets.ImageFolder(root='/train-data/ILSVRC2012/val', transform=transform)

    BATCH_SIZE = params.batch_size
    NUM_WORKERS = params.num_workers

    print("Image Shape: {}".format(train_dataset[0][0].numpy().shape), end = '\n\n')
    print("Training Set:   {} samples".format(len(train_dataset)))
    print("Testing Set:       {} samples".format(len(test_dataset)))

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, sampler=RandomSampler(train_dataset, replacement=True, num_samples=1000))
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, num_workers=NUM_WORKERS, sampler=RandomSampler(test_dataset, replacement=True, num_samples=100))

    return train_loader, test_loader

print("Laoding dataset....")
if params.dataset=="cifar10":
    train_loader, test_loader  = load_cifar()
elif params.dataset=="imagenet":
    train_loader, test_loader = load_imagenet()
else:
    print("dataset not find!")

# Train the model
print("Experiment - model version: {}".format(params.model_version))

model = Model_ComDefend.comdefend(noise=False, inference=False)
model = model.cuda()

trainer = Trainer(model, params)
trainer.train_and_evaluate(train_loader, test_loader)
