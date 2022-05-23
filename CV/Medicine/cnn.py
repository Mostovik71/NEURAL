import pandas as pd
import torch.nn as nn
import torch
import cv2
from torchvision import datasets
import torchvision
import torchvision.transforms as transforms
from sklearn.model_selection import train_test_split
import numpy as np
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os


class ConvNet(nn.Module):
    def __init__(self):
        super(ConvNet, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2))
        self.layer2 = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2))

        self.drop_out = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(802816, 2)

        self.softmax = F.softmax

    def forward(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(x)
        out1 = out1.reshape(out1.size(0), -1)
        out2 = out2.reshape(out2.size(0), -1)

        out = torch.cat((out1, out2), dim=1)
        # print(out.shape)
        # out = out.reshape(out.size(0), -1)
        out = self.drop_out(out)
        out = self.fc1(out)
        out = self.softmax(out)
        return out


def img_resize(image):
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    image = cv2.resize(image, (227, 227)).reshape((3, 227, 227))
    image = torchvision.transforms.functional.to_tensor(image)
    image = image.reshape((3, 227, 227))
    image = image.unsqueeze(0)
    return image


no = []
yes = []
pathno = 'C:/Users/mosto/PycharmProjects/Movmed/Task4/DATA/NO/'
pathyes = 'C:/Users/mosto/PycharmProjects/Movmed/Task4/DATA/YES/'
itemsno = os.listdir(pathno)
itemsyes = os.listdir(pathyes)

for i in itemsno:
    i = pathno + i
    image = cv2.imread(i)
    image = img_resize(image)
    no.append(image)
for i in itemsyes:
    i = pathyes + i
    image = cv2.imread(i)
    image = img_resize(image)
    yes.append(image)
labelsyes = [1 for i in range(len(yes))]
labelsno = [0 for i in range(len(no))]
labelsno.extend(labelsyes)
labels = labelsno
yes.extend(no)
features = yes
# X_train, X_valid, y_train, y_valid = train_test_split(features, labels, test_size=0.2, random_state=75)

# print((train[0]))
yes = [yes[i][0] for i in range(len(yes))]
no = [no[i][0] for i in range(len(no))]
n = [(i, k) for i, k in zip(no, labelsno)]
y = [(i, k) for i, k in zip(yes, labelsyes)]
y.extend(n)

batch_size = 4
learning_rate = 0.000015
epochs = 20

train, valid = train_test_split(y, test_size=0.3, random_state=74)
train_loader = DataLoader(train, batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(valid, batch_size=batch_size, shuffle=True)

model = ConvNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
# for epoch in range(epochs):  # Цикл по эпохе
#     currloss = 0.0
#     for i, (images, labels) in enumerate(train_loader):  # Цикл по батчам
#
#         outputs = model(images)
#
#         loss = criterion(outputs, labels)
#
#         optimizer.zero_grad()
#
#         loss.backward()
#         optimizer.step()
#         currloss+=loss.item()
#     print(currloss)
# torch.save(model, 'model.pth')
model = torch.load('model.pth')
model.eval()
p = []
l = []
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in valid_loader:
        outputs = model(images)

        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        p.extend(predicted.tolist())
        l.extend(labels.tolist())
        correct += (predicted == labels).sum().item()

    print(p)
    print(l)
    print(l.count(0))
    print('Test Accuracy of the model on the test images: {}'.format((correct / total)))
