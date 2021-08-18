import torch
from torch.autograd import Variable
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import os
from fc import Classifier
from tqdm import tqdm
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
batch_size=100
test_losses = []
DEVICE='cuda:0'
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True)
criterion = nn.NLLLoss()
model=torch.load('modelmnist.pth')
test_loss = 0
accuracy = 0
with torch.no_grad():
    model.eval()
    for images, labels in tqdm(test_loader):
        images = images.view(100, 784)
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        log_ps = model(images)
        test_loss += criterion(log_ps, labels)

        ps = torch.exp(log_ps)  # Преобразование к вероятностям

        top_p, top_class = ps.topk(1, dim=1)  # С какой уверенностью и что предсказала модель
        equals = top_class == labels.view(*top_class.shape)  # Перевод лейблов в тот же формат что и предсказания, выход - True and False
        accuracy += torch.mean(equals.type(torch.FloatTensor))
        stop=1
test_losses.append(test_loss / len(test_loader))
print("Test Loss: {:.3f}.. ".format(test_losses[-1]),"Test Accuracy: {:.3f}".format(accuracy / len(test_loader)))

