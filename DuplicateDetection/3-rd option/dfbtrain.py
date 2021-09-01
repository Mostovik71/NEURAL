import torch
from torch.autograd import Variable
import torch.nn as nn
import pandas as pd
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
import os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

class Classifier(nn.Module):
    def __init__(self):
        super().__init__()
        # 5 Hidden Layer Network
        self.fc1 = nn.Linear(1536, 768)
        self.fc2 = nn.Linear(768, 384)
        self.fc3 = nn.Linear(384, 196)
        self.fc4 = nn.Linear(196, 98)
        self.fc5 = nn.Linear(98, 49)
        self.fc6 = nn.Linear(49, 2)




        self.log_softmax = F.log_softmax

    def forward(self, x):
        x = F.tanh(self.fc1(x))
        x = F.sigmoid(self.fc2(x))
        x = F.tanh(self.fc3(x))
        x = F.sigmoid(self.fc4(x))
        x = F.tanh(self.fc5(x))
        x = self.log_softmax(self.fc6(x), dim=1)

        return x


batch_size = 4
train_data = pd.read_excel('forlog1024.xlsx')
train_data.drop('Unnamed: 0', axis=1, inplace=True)
#train_data.rename(columns = {'0.1':'is_duplicate'}, inplace = True)
#train_data.drop('0.1', axis=1, inplace=True)
y=train_data['is_duplicate']
train_data.drop('is_duplicate', axis=1, inplace=True)
X=train_data
X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=33)
class MyDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.Tensor(X)
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return (self.X[index], self.y[index])

train=MyDataset(X_train.values, y_train.values)
valid=MyDataset(X_valid.values, y_valid.values)
train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=True)


DEVICE='cuda:0'
model = Classifier()
model.to(DEVICE)

criterion = nn.NLLLoss()
from transformers import AdamW
optimizer = AdamW(model.parameters(),
                  lr = 2e-5,
                  eps = 1e-8
                )
epochs = 9

import time
import datetime

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''

    elapsed_rounded = int(round((elapsed)))


    return str(datetime.timedelta(seconds=elapsed_rounded))

def fit_batch(dataloader, model, optimizer, epoch):
    total_train_loss = 0

    for batch, labels in tqdm(dataloader, desc=f"Training epoch:{epoch}", unit="batch"):
        batch = batch.to(DEVICE)
        labels = labels.to(DEVICE, dtype=torch.long)


        optimizer.zero_grad()

        log_ps = model(batch)
        loss = criterion(log_ps, labels)

        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    return total_train_loss


import numpy

from sklearn.metrics import accuracy_score

def eval_batch(dataloader, model, metric=accuracy_score):
    total_eval_accuracy = 0
    total_eval_loss = 0
    predictions, predicted_labels = [], []

    for batch, labels in tqdm(dataloader, desc="Evaluating", unit="batch"):

        batch = batch.to(DEVICE)
        labels = labels.to(DEVICE, dtype=torch.long)
        with torch.no_grad():
            log_ps = model(batch)
            loss=criterion(log_ps, labels)





        total_eval_loss += loss

        ps = log_ps

        top_p, top_class = ps.topk(1, dim=1)
        equals = top_class == labels.view(*top_class.shape)
        total_eval_accuracy += torch.mean(equals.type(torch.FloatTensor))
        predictions=top_p
        predicted_labels=top_class
    return total_eval_accuracy, total_eval_loss, predictions, predicted_labels




import random


seed_val = 42
random.seed(seed_val)
numpy.random.seed(seed_val)
torch.manual_seed(seed_val)


def train(train_dataloader, validation_dataloader, model, optimizer, epochs):
    
    training_stats = []

  
    total_t0 = time.time()

    for epoch in range(0, epochs):

        t0 = time.time()


        total_train_loss = 0


        model.train()

        total_train_loss = fit_batch(train_dataloader, model, optimizer, epoch)


        avg_train_loss = total_train_loss / len(train_dataloader)


        training_time = format_time(time.time() - t0)

        t0 = time.time()


        model.eval()

        total_eval_accuracy, total_eval_loss, _, _ = eval_batch(validation_dataloader, model)
        FILE = 'modelfeedforward1024.pth'
        torch.save(model, FILE)

        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        avg_train_loss = total_train_loss / len(train_dataloader)
        print(f"  Accuracy: {avg_val_accuracy}")

        
        avg_val_loss = total_eval_loss / len(validation_dataloader)


        
        validation_time = format_time(time.time() - t0)
        print(f"  Train Loss: {avg_train_loss}")
        print(f"  Validation Loss: {avg_val_loss}")

        
        training_stats.append(
            {
                'epoch': epoch,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")

    print(f"Total training took {format_time(time.time() - total_t0)}")
    return training_stats
training_stats = train(train_loader, valid_loader, model, optimizer, epochs)










'''
for e in range(epochs):
    running_loss = 0
    for batch, labels in train_loader:
        batch=batch.to(DEVICE)
        labels=labels.to(DEVICE, dtype=torch.long)
        steps += 1
        # Prevent accumulation of gradients
        optimizer.zero_grad()
        # Make predictions
        log_ps = model(batch)
        loss = criterion(log_ps, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        if steps % print_every == 0:
            test_loss = 0
            accuracy = 0


            with torch.no_grad():
                model.eval()
                for batch, labels in valid_loader:
                    batch = batch.to(DEVICE)
                    labels=labels.to(DEVICE, dtype=torch.long)
                    log_ps = model(batch)
                    test_loss += criterion(log_ps, labels)

                    ps=log_ps

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
'''
