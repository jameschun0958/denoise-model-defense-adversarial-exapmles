from __future__ import print_function

import os
import time
import numpy as np
import math

import torch
from torch import nn, optim
from torch.autograd import Variable
from torch.optim.lr_scheduler import StepLR

from utils import progress_bar

print("==> Check devices..")
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Current device: ",device)
print("Our selected device: ", torch.cuda.current_device())
print(torch.cuda.device_count(), " GPUs is available")

class Trainer(object):
    def __init__(self,
                 model,
                 params):
        self.model = model
        self.params = params
        self.model_name = params.model_name

        self.best_loss = 100
        self.best_epoch = 0


    def train(self, epoch, no_of_steps, trainloader, lr):
        self.model.train()
        lambd = 0.0001
        train_loss, correct, total = 0, 0, 0

        # Declare optimizer.
        if not hasattr(self, 'optimizer'):
            self.optimizer = optim.Adam(self.model.parameters(), lr)
        
        self.adjust_learning_rate(self.optimizer, epoch, lr)

        # Loss criterion
        criterion = nn.MSELoss()

        for idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)

            linear_code, binary_code, outputs = self.model(inputs)

            loss = criterion(outputs, inputs) + lambd * criterion(torch.zeros(binary_code.size()).to(device), binary_code)

            # Calculate the gradients
            self.optimizer.zero_grad()
            loss.backward()
            
            self.optimizer.step()
            
            train_loss += loss.item()

            #_, predicted = outputs.max(1)

            #total += targets.size(0)
            #correct += (targets == predicted).sum().item()

            progress_bar(
                idx, len(trainloader), 'Loss: %.6f | PSNR: %d' %
                (train_loss / (idx + 1), self.psnr(inputs, outputs)))

    
    def evaluate(self, epoch, testloader):
        self.model.eval()

        test_loss = 0
        correct = 0
        total = 0
        lambd = 0.0001

        criterion = nn.MSELoss()

        with torch.no_grad():
            for idx, (test_x, test_y) in enumerate(testloader):
                test_x, test_y = test_x.to(device), test_y.to(device)

                linear_code, binary_code, outputs = self.model(test_x)

                loss = criterion(outputs, test_x) + lambd * criterion(binary_code, torch.zeros(binary_code.size()).to(device))

                test_loss += loss.item()

                #_, predicted = outputs.max(1)
                
                #total += test_y.size(0)
                #correct += (predicted == test_y).sum().item()

                progress_bar(
                    idx, len(testloader), 'Loss: %.6f | PSNR: %.d' %
                    (test_loss / (idx + 1), self.psnr(test_x, outputs)))

        # acc = 100.0 * correct / total

        loss = test_loss / (idx + 1)

        if loss < self.best_loss:
            self.save_model(self.model, self.model_name, loss, epoch)

    def evaluateTopk(self, path, testloader, k=1):
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['net'])
        self.model.eval()

        test_loss = 0
        correct = 0
        total = 0

        criterion = nn.CrossEntropyLoss()

        with torch.no_grad():
            for idx, (test_x, test_y) in enumerate(testloader):
                test_x, test_y = test_x.to(device), test_y.to(device)

                outputs = self.model(test_x)
                loss = criterion(outputs, test_y)

                test_loss += loss.item()

                maxk = max((1,k))
                
                total += test_y.size(0)

                test_y = test_y.view(-1, 1)
                _, pred = outputs.topk(maxk, 1, True, True)
                correct += torch.eq(pred, test_y).sum().item()

                progress_bar(
                    idx, len(testloader), 'Loss: %.6f | Acc: %.3f%% (%d/%d)' %
                    (loss / (idx + 1), 100. * correct / total, correct, total))

    def adjust_learning_rate(self, optimizer, epoch, lr):
        """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
        lr = float(lr * (0.1 ** (epoch // 10)))
        print('Learning rate:{}'.format(lr))
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

    def save_model(self, model, model_name, loss, epoch):
        save_name = os.path.join('experiments', model_name, 'weights', 'CIFAR10-ComDefend.pt')

        if not os.path.exists(os.path.dirname(save_name)):
            os.makedirs(os.path.dirname(save_name))

        torch.save(model.state_dict(), save_name)
        print("\nSaved state at %.6f loss. Prev loss: %.6f" %
              (loss, self.best_loss))
        self.best_loss = loss
        self.best_epoch = epoch

    def psnr(self, im1,im2):
        mse = torch.mean( (im1 - im2) ** 2 )
        if mse < 1.0e-10:
            return 100
        PIXEL_MAX = 1
        return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))

    def train_and_evaluate(self, traindataloader, testdataloader, restore_file=None):

        no_of_steps = self.params.num_epochs
        lr = self.params.learning_rate

        for epoch in range(no_of_steps):
            print('\nEpoch: %d' % (epoch + 1))
            self.train(epoch, no_of_steps, traindataloader, lr)
            self.evaluate(epoch, testdataloader)
