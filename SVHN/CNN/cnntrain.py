import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torchvision import datasets
from sklearn.model_selection import train_test_split
import numpy as np


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()#Входное изображение  - 32х32 пикселя
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=1, stride=2))#Выход после 1 сверточного слоя - 32х32 пикселей, 32 канала(16х16х32)
        self.layer2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2))#Выход после 2 сверточного слоя - 8х8 пикселей, 64 канала(8х8х64)
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=5, stride=1, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=2, stride=2))


        self.drop_out = nn.Dropout()#Отсеивающий слой для обхода переобучения
        self.fc1 = nn.Linear(4 * 4 * 256, 1000)
        self.fc2 = nn.Linear(1000, 10)


    def forward(self, x):# Функция прямого распространения, для прохода данных через сеть
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = out.reshape(out.size(0), -1)#Преобразует данные из 8х8х64 в 4096х1
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.fc2(out)

        return out# Модель возвращает числа перед SoftMax

DEVICE='cuda:0'
num_epochs = 6
num_classes = 10
batch_size = 100
learning_rate = 0.001
train_data=datasets.SVHN('../data', 'train', download=True,
                   transform=transforms.Compose([  # transforms.Compose - позволяет делать преобразования в датасете
                       transforms.ToTensor(),  # .ToTensor - преобразует датасет в PyTorch тензор
                       transforms.Normalize((0.1307,), (0.3081,))
                       # Нормализует данные(в диапазон от 0 до 1), 0.1307 и 0.3081 - среднее и стандартное отклонение датасета, значения нужно установить для каждого канала
                   ]))
train, valid = train_test_split(train_data, test_size=0.2, random_state=42)
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=True)
model = ConvNet()
model.to(DEVICE)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)


total_step = len(train_loader)
loss_list = []
acc_list = []
for epoch in range(num_epochs):#Цикл по эпохе
    for i, (images, labels) in enumerate(train_loader):#Цикл по батчам
        # Run the forward pass
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss_list.append(loss.item())

        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

       
        total = labels.size(0)
        _, predicted = torch.max(outputs.data, 1)#максимальные предсказанные вероятности и их индексы(predicted - число, которое предсказала модель)
        correct = (predicted == labels).sum().item()#Сколько из батча модель предсказала правильно
        acc_list.append(correct / total)#Accuracy батча

        if (i + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}, Accuracy: {:.2f}%'
                  .format(epoch + 1, num_epochs, i + 1, total_step, loss.item(),
                          (correct / total) * 100))
        stop = 1

model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in valid_loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    print('Test Accuracy of the model on the ' + str(total) + ' test images: {} %'.format((correct / total) * 100))


#torch.save(model,'cnnmodel.pth')

