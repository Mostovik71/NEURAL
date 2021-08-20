import torch
from torch.autograd import Variable
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import os
from sklearn.model_selection import train_test_split
from torchvision import datasets, transforms
from tqdm import tqdm
import matplotlib.pyplot as plt
#АБСОЛЮТНО БЕСПОЛЕЗНАЯ ВЕЩЬ, КОТОРАЯ ЕЩЕ И ОШИБКИ СМЕЕТ ВЫДАВАТЬ (нужно подстраивать размер батча под кратность вал. выборки)
img_size = 32
number_of_channels = 3
class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 5 Hidden Layer Network
        self.fc1 = nn.Linear((img_size**2) * number_of_channels, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)
        self.fc5 = nn.Linear(64, 10)


        self.dropout = nn.Dropout(p=0.2)

        self.log_softmax = F.log_softmax

    def forward(self, x):
        x = x.view(x.size(0), -1)

        x = self.dropout(F.relu(self.fc1(x)))

        x = self.dropout(F.relu(self.fc2(x)))

        x = self.dropout(F.relu(self.fc3(x)))

        x = self.dropout(F.relu(self.fc4(x)))


        x = self.log_softmax(self.fc5(x), dim=1)

        return x

batch_size = 100


train_data=datasets.SVHN('../data', 'train', download=True,
                   transform=transforms.Compose([  # transforms.Compose - позволяет делать преобразования в датасете

                       transforms.ToTensor(),  # .ToTensor - преобразует датасет в PyTorch тензор
                       transforms.Normalize((0.1307,), (0.3081,))
                       # Нормализует данные(в диапазон от 0 до 1), 0.1307 и 0.3081 - среднее и стандартное отклонение датасета, значения нужно установить для каждого канала
                   ]))
valid_data=datasets.SVHN('../data', 'test', download=True,
                   transform=transforms.Compose([  # transforms.Compose - позволяет делать преобразования в датасете

                       transforms.ToTensor(),  # .ToTensor - преобразует датасет в PyTorch тензор
                       transforms.Normalize((0.1307,), (0.3081,))
                       # Нормализует данные(в диапазон от 0 до 1), 0.1307 и 0.3081 - среднее и стандартное отклонение датасета, значения нужно установить для каждого канала
                   ]))

#train, valid = train_test_split(train_data, test_size=0.2, random_state=42)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=16, shuffle=True)


DEVICE='cuda:0'

model = Classifier()
model.to(DEVICE)
criterion = nn.NLLLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0015)
epochs = 1
steps = 0
print_every = 50
train_losses, test_losses = [], []
for e in range(epochs):
    running_loss = 0
    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        steps += 1

        optimizer.zero_grad()


        log_ps = model(images)
        loss = criterion(log_ps, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0


            with torch.no_grad():
                model.eval()
                for images1, labels in tqdm(valid_loader):

                    images1 = images1.to(DEVICE)
                    labels = labels.to(DEVICE)
                    log_ps = model(images)

                    test_loss = criterion(log_ps, labels)

                    ps = torch.exp(log_ps)#Преобразование к вероятностям

                    top_p, top_class = ps.topk(1, dim=1)#С какой уверенностью и что предсказала модель
                    equals = top_class == labels.view(*top_class.shape)#Перевод лейблов в тот же формат что и предсказания
                    accuracy += torch.mean(equals.type(torch.FloatTensor))

            model.train()

            train_losses.append(running_loss / len(train_loader))
            test_losses.append(test_loss / len(valid_loader))

            print("Epoch: {}/{}.. ".format(e + 1, epochs),
                  "Training Loss: {:.3f}.. ".format(train_losses[-1]),
                  "Test Loss: {:.3f}.. ".format(test_losses[-1]),
                  "Test Accuracy: {:.3f}".format(accuracy / len(valid_loader)))
#torch.save(model,'modelmnist.pth')


