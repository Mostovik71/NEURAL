from cnntrain import ConvNet
import torch
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torchvision import datasets, transforms
test_data=datasets.SVHN('../data', 'test', download=True,
                   transform=transforms.Compose([  # transforms.Compose - позволяет делать преобразования в датасете
                       transforms.ToTensor(),  # .ToTensor - преобразует датасет в PyTorch тензор
                       transforms.Normalize((0.1307,), (0.3081,))
                       # Нормализует данные(в диапазон от 0 до 1), 0.1307 и 0.3081 - среднее и стандартное отклонение датасета, значения нужно установить для каждого канала
                   ]))
test_loader=torch.utils.data.DataLoader(test_data, batch_size = 100, shuffle = True)
import matplotlib.pyplot as plt
batch_size = 100
test_losses = []
DEVICE = 'cuda:0'
criterion = nn.NLLLoss()
model=torch.load('cnnsvhn.pth')
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
    print('Test Accuracy of the model on the ' + str(total) + ' test images: {} %'.format((correct / total) * 100))
