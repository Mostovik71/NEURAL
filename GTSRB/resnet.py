import numpy as np
import seaborn as sns
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os
import gc
import cv2
import matplotlib.pyplot as plt
from torchvision import transforms,datasets,models
from torch.utils.data import Dataset,DataLoader
from sklearn.model_selection import GroupKFold
from sklearn.metrics import accuracy_score,roc_auc_score
from sklearn.preprocessing import StandardScaler
import time
import datetime
from PIL import Image
import warnings
from tqdm import tqdm
import random
import pandas as pd

warnings.simplefilter('ignore')
torch.manual_seed(7)
np.random.seed(7)
random.seed(7)
torch.cuda.manual_seed(7)
torch.backends.cudnn.deterministic = True
DEVICE='cuda:0'
path = 'data/Train'
image_path=[]
target=[]
for i in os.listdir(path):

    for j in os.listdir(os.path.join(path,i)):
        image_path.append(os.path.join(path,i,j))
        target.append(i)
table = {'image_path': image_path, 'target': target}
df = pd.DataFrame(data=table)
df = df.sample(frac = 1).reset_index(drop=True)
test_df = pd.read_csv('data/Test.csv')

class CustomDataset(Dataset):
    def __init__(self, dataframe,transform):
        self.dataframe = dataframe
        self.transform = transform
    def __len__(self):
        return self.dataframe.shape[0]
    def __getitem__(self,index):
        image = self.dataframe.iloc[index]['image_path']
        image = cv2.imread(image)
        image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        label = int(self.dataframe.iloc[index]["target"])
        return {"image": torch.tensor(image, dtype=torch.float), "targets": torch.tensor(label, dtype=torch.long)}

class test_CustomDataset(Dataset):
    def __init__(self,dataframe,transform):
        self.dataframe = dataframe
        self.transform = transform
    def __len__(self):
        return self.dataframe.shape[0]
    def __getitem__(self,index):
        image = self.dataframe.iloc[index]['Path']
        image = os.path.join('data',image)
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(image)
        if self.transform:
            image = self.transform(image)
        label = int(self.dataframe.iloc[index]["ClassId"])
        return {"image": torch.tensor(image, dtype=torch.float), "targets": torch.tensor(label, dtype = torch.long)}
stop=1
def get_model(classes=43):
    model = models.resnet34(pretrained=True)
    features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.BatchNorm1d(features),
        nn.Dropout(p=0.25),
        nn.Linear(in_features = features, out_features = 2048),
        nn.ReLU(),
        nn.BatchNorm1d(2048),
        nn.Dropout(p=0.5),
        nn.Linear(in_features = 2048, out_features = classes)
    )
    return model
model = get_model()
model.to(DEVICE)

train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])
test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
])

adam_optimizer = optim.Adam(model.parameters(),lr = 0.00003)
loss_function = nn.CrossEntropyLoss()
train_dataset = CustomDataset(
dataframe=df,
transform=train_transform)
train_loader = DataLoader(train_dataset, batch_size = 16, shuffle = True, num_workers = 4)

valid_dataset = test_CustomDataset(
dataframe=test_df,
transform=test_transform)
valid_loader = DataLoader(valid_dataset, batch_size=16, shuffle=False, num_workers=4)
stop=1
'''
def model_train():
 for epochs in tqdm(range(10), desc="Epochs"):
    model.train()
    for data_in_model in tqdm(train_loader, desc="Training"):
        inputs = data_in_model['image']
        target = data_in_model['targets']

        inputs = inputs.to(DEVICE, dtype=torch.float)
        targets = target.to(DEVICE, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        loss.backward()
        optimizer.step()

    model.eval()
    final_targets = []
    final_outputs = []
    val_loss = 0
    with torch.no_grad():
        for data_in_model in tqdm(valid_loader, desc="Evaluating"):
            inputs = data_in_model['image']
            targets = data_in_model['targets']

            inputs = inputs.to(DEVICE, dtype=torch.float)
            targets = targets.to(DEVICE, dtype=torch.long)

            outputs = model(inputs)
            loss = loss_function(outputs, targets)
            val_loss += loss
            _, predictions = torch.max(outputs, 1)

            targets = targets.detach().cpu().numpy().tolist()
            predictions = predictions.detach().cpu().numpy().tolist()

            final_targets.extend(targets)
            final_outputs.extend(predictions)
    PREDS = np.array(final_outputs)
    TARGETS = np.array(final_targets)
    acc = (PREDS == TARGETS).mean() * 100
    #print("EPOCH: {}/10".format(epochs + 1))
    #print("ACCURACY---------------------------------------------------->{}".format(acc))
    #print("LOSS-------------------------------------------------------->{}".format(val_loss))
'''

from transformers import get_linear_schedule_with_warmup

epochs = 4
total_steps = len(train_loader) * epochs

scheduler = get_linear_schedule_with_warmup(adam_optimizer,
                                            num_warmup_steps = 0,
                                            num_training_steps = total_steps)
import time
import datetime


def format_time(elapsed):

    elapsed_rounded = int(round((elapsed)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def fit_batch(dataloader, model, optimizer, epoch):
    total_train_loss = 0

    for batch in tqdm(dataloader, desc=f"Training epoch:{epoch}", unit="batch"):

        inputs = batch['image']
        target = batch['targets']

        model.zero_grad()
        inputs = inputs.to(DEVICE, dtype=torch.float)
        targets = target.to(DEVICE, dtype=torch.long)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)

        total_train_loss += loss
        #total_train_accuracy +=
        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

        optimizer.step()

        scheduler.step()

    return total_train_loss


import numpy

from sklearn.metrics import accuracy_score


def eval_batch(dataloader, model, metric=accuracy_score):
    total_eval_accuracy = 0
    total_eval_loss = 0
    final_targets, final_outputs = [], []

    for batch in tqdm(dataloader, desc="Evaluating", unit="batch"):
        
        inputs = batch['image']
        targets = batch['targets']

        inputs = inputs.to(DEVICE, dtype=torch.float)
        targets = targets.to(DEVICE, dtype=torch.long)
        #model.cuda()
        outputs = model(inputs)
        loss = loss_function(outputs, targets)
        total_eval_loss += loss
        _, predictions = torch.max(outputs, 1)

        targets = targets.detach().cpu().numpy().tolist()
        predictions = predictions.detach().cpu().numpy().tolist()

        final_targets.extend(targets)
        final_outputs.extend(predictions)

    return  total_eval_loss, final_targets, final_outputs


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
        FILE = 'roadsignskaggle.pth'
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



if __name__=='__main__':
    train(train_loader, valid_loader, model, adam_optimizer, epochs)
