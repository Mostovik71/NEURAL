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

model=torch.load('modelfeedforward1024.pth')
batch_size = 1
valid_data = pd.read_excel('embeddings.xlsx')
data=pd.read_excel('')
valid_data['is_duplicate']=data['is_duplicate']
valid_data.drop('Unnamed: 0', axis=1, inplace=True)
#valid_data.rename(columns = {'0.1':'is_duplicate'}, inplace = True)

y=valid_data['is_duplicate']
valid_data.drop('is_duplicate', axis=1,inplace=True)
X=valid_data

criterion = nn.NLLLoss()
X.reset_index(drop=True,inplace=True)


class MyDataset(Dataset):

    def __init__(self, X, y):
        self.X = torch.Tensor(X)
        self.y = torch.from_numpy(y).float()

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        return (self.X[index], self.y[index])

valid=MyDataset(X.values, y.values)


valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size)
DEVICE='cuda:0'

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
        #predictions.append(top_p.tolist()[0][0])
        predictions.append(log_ps)
        predicted_labels.append(top_class.tolist()[0][0])

    return total_eval_accuracy, total_eval_loss, predictions, predicted_labels
model.eval()
_, _,predictions,predicted_labels = eval_batch(valid_loader, model)
dataframe=pd.concat([data[['...','...','is_duplicate']],pd.Series(predicted_labels)],axis=1)
dataframe.to_excel('feedforward1024.xlsx')
