import pandas as pd
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import AdamW
import warnings
from sklearn.preprocessing import Normalizer
from tqdm import tqdm
from torchvision import datasets, transforms
import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

X = pd.read_excel('Mostovik_all_new.xlsx')
# X.drop('Unnamed: 0', inplace=True, axis=1)
y = X['label']
X.drop('label', inplace=True, axis=1)
# transformer = Normalizer().fit(X)
# X = pd.DataFrame(transformer.transform(X))

X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=75)

batch_size = 12
epochs = 20


class Classifier(nn.Module):
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(16, 6)
        self.fc2 = nn.Linear(6, 2)
        self.dropout = nn.Dropout(p=0.2)
        self.softmax = F.softmax

    def forward(self, x):
        x = (F.sigmoid(self.fc1(x)))
        x = self.softmax(self.fc2(x), dim=1)

        return x


class MyDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.Tensor(X)
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return (self.X[index], self.y[index])


train = MyDataset(X_train.values, y_train.values)
valid = MyDataset(X_valid.values, y_valid.values)
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=True)

model = Classifier()
loss_fnc = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0015)

model.train()
losses = []
for epoch in range(epochs):
    # Print epoch
    print(f'Starting epoch {epoch + 1}')

    # Set current loss value
    current_loss = 0.0

    # Iterate over the DataLoader for training data
    for inputs, targets in train_loader:
        targets = targets.to(dtype=torch.long)
        # Zero the gradients
        optimizer.zero_grad()

        # Perform forward pass
        outputs = model(inputs)

        # Compute loss
        loss = loss_fnc(outputs, targets)

        # Perform backward pass
        loss.backward()

        # Perform optimization
        optimizer.step()

        # Print statistics
        current_loss += loss.item()
    losses.append(current_loss)
    # print('Train loss:', current_loss/len(train_loader))

    test_loss = 0
    accuracy = 0

    # Turn off gradients for validation
    with torch.no_grad():
        model.eval()
        for inputs, targets in train_loader:
            targets = targets.to(dtype=torch.long)
            log_ps = model(inputs)
            test_loss += loss_fnc(log_ps, targets)

            ps = (log_ps)

            top_p, top_class = ps.topk(1)
            equals = top_class == targets.view(
                *top_class.shape)  # Перевод лейблов в тот же формат что и предсказания
            accuracy += torch.mean(equals.type(torch.FloatTensor))
    print("Epoch: {}/{}.. ".format(epoch + 1, epochs),
          "Training Loss: {:.3f}.. ".format(current_loss / len(train_loader)),
          "Test Loss: {:.3f}.. ".format(test_loss / len(valid_loader)),
          "Test Accuracy: {:.3f}".format(accuracy / len(train_loader)))
torch.save(model, 'model.pth')


model = torch.load('model.pth')
model.eval()
batch_size = 4
preds = pd.DataFrame([])

for batch, labels in tqdm(valid_loader, desc="Evaluating", unit="batch"):
    labels = labels.to(dtype=torch.long)
    with torch.no_grad():
        log_ps = model(batch)

    top_p, top_class = log_ps.topk(1)
    # equals = top_class == labels.view(*top_class.shape)
    # total_eval_accuracy += torch.mean(equals.type(torch.FloatTensor))
    top_class = torch.flatten(top_class)

    preds = preds.append(
        pd.concat([pd.Series(top_class.tolist(), name='Predictions'), pd.Series(labels.tolist(), name='Labels')],
                  axis=1))

# print(((preds['Predictions'] == preds['Labels']) == 1).value_counts())
# print((preds['Predictions'] == (preds['Labels'] == 0)).value_counts())
labels = preds['Labels']
preds = preds['Predictions']
labels.reset_index(drop=True, inplace=True)
preds.reset_index(drop=True, inplace=True)
#print((preds == labels).value_counts())

ones = labels[labels == 1]
zeros = labels[labels == 0]
print((preds == labels).value_counts(), 'Accuracy')
print(pd.Series(preds[zeros.index]).value_counts(), 'Specificity')
print(pd.Series(preds[ones.index]).value_counts(), 'Sensitivity')
