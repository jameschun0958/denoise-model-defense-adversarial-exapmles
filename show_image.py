import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split,TensorDataset
import torchvision
from torchvision import transforms, datasets
import torchvision.models as models

from models import WideResNet
from models import resnet
from models import Model_ComDefend

import os
import numpy as np
import matplotlib.pyplot as plt

import torchattacks
from PIL import Image
from torchvision.utils import save_image

def load_cifar():
    transform = transforms.Compose([transforms.ToTensor()])
            
    test_dataset = datasets.CIFAR10('./data', train=False, download=True, transform=transform)
    
    print("Image Shape: {}".format(test_dataset[0][0].numpy().shape), end = '\n\n')
    print("Testing Set:       {} samples".format(len(test_dataset)))

    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=32)
    
    return test_loader

def imshow(img, title):
    npimg = img.numpy()
    plt.title(title)
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()
    
print("Laoding dataset....")
test_loader  = load_cifar()

test_model = WideResNet.WideResNet(depth=40, widen_factor=4, num_classes=10)
checkpoint = torch.load("./models/pretrained_weight/CIAFR10-WRN40-4.pt")
test_model.load_state_dict(checkpoint['net'])
test_model.eval()

# Test and plot image
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

# get some random training images
dataiter = iter(test_loader)
images, labels = dataiter.next()

# show images
imshow(torchvision.utils.make_grid(images),title="clean image")

atk = torchattacks.GN(test_model, sigma=0.1) # Add Gaussian Noise
adv_images = atk(images, labels)
# show images
imshow(torchvision.utils.make_grid(adv_images),title="noise image")

model = Model_ComDefend.comdefend(noise=False, inference=True)
model.load_state_dict(torch.load("./experiments/CIFAR10-ComDefend/weights/CIFAR10-ComDefend.pt"))
model.eval()
linear_code, binary_code, outputs = model(images)
outputs = outputs.detach()

# show images
imshow(torchvision.utils.make_grid(outputs),title="denoise image")