import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms
from cnntrain import ConvNet
import matplotlib.pyplot as plt
batch_size = 100
test_losses = []
DEVICE = 'cuda:0'
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('../data', train=False, transform=transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])),
    batch_size=batch_size, shuffle=True)
criterion = nn.NLLLoss()
model=torch.load('cnnmodel.pth')
test_loss = 0
accuracy = 0
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the ' +str(total)+ ' test images: {} %'.format((correct / total) * 100))
