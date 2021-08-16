import numpy as np
import pandas as pd
from sklearn import metrics
import transformers
import torch
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertModel, BertConfig
import torch.nn as nn
from torch import optim
import torch.nn.functional as F

df=pd.read_excel('newnodupl.xlsx')
DEVICE='cuda:0'

class SiameseNetworkDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.question1 = dataframe.Descr1
        self.question2 = dataframe.Descr2
        self.targets = dataframe.is_duplicate
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def tokenize(self, input_text):
        input_text = " ".join(input_text.split())

        inputs = self.tokenizer.encode_plus(
            input_text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )
        ids = inputs['input_ids']
        mask = inputs['attention_mask']
        token_type_ids = inputs["token_type_ids"]
        return ids, mask, token_type_ids

    def __getitem__(self, index):
        ids1, mask1, token_type_ids1 = self.tokenize(str(self.question1[index]))
        ids2, mask2, token_type_ids2 = self.tokenize(str(self.question2[index]))

        return {
            'ids': [torch.tensor(ids1, dtype=torch.long), torch.tensor(ids2, dtype=torch.long)],
            'mask': [torch.tensor(mask1, dtype=torch.long), torch.tensor(mask2, dtype=torch.long)],
            'token_type_ids': [torch.tensor(token_type_ids1, dtype=torch.long),
                               torch.tensor(token_type_ids2, dtype=torch.long)],
            'targets': torch.tensor(self.targets[index], dtype=torch.float)
        }

class TwinBert(nn.Module):
        def __init__(self):
            super(TwinBert, self).__init__()
            self.model = transformers.BertModel.from_pretrained('DeepPavlov/rubert-base-cased')

        def forward_once(self, ids, mask, token_type_ids):
            output = self.model(ids, attention_mask=mask, token_type_ids=token_type_ids)[1]
            return output

        def forward(self, ids, mask, token_type_ids):
            output1 = self.forward_once(ids[0], mask[0], token_type_ids[0])
            output2 = self.forward_once(ids[1], mask[1], token_type_ids[1])
            return output1, output2

model = TwinBert()
model.to(DEVICE)
MAX_LEN = 512
TRAIN_BATCH_SIZE = 4
VALID_BATCH_SIZE = 4
EPOCHS = 2
LEARNING_RATE = 1e-05
tokenizer = BertTokenizer.from_pretrained('DeepPavlov/rubert-base-cased')
train_size = 0.8
train_dataset=df.sample(frac=train_size,random_state=200).reset_index(drop=True)
test_dataset=df.drop(train_dataset.index).reset_index(drop=True)


print("FULL Dataset: {}".format(df.shape))
print("TRAIN Dataset: {}".format(train_dataset.shape))
print("TEST Dataset: {}".format(test_dataset.shape))

training_set = SiameseNetworkDataset(train_dataset, tokenizer, MAX_LEN)
testing_set = SiameseNetworkDataset(test_dataset, tokenizer, MAX_LEN)
train_params = {'batch_size': TRAIN_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

test_params = {'batch_size': VALID_BATCH_SIZE,
                'shuffle': True,
                'num_workers': 0
                }

training_loader = DataLoader(training_set, **train_params)
testing_loader = DataLoader(testing_set, **test_params)
class CosineContrastiveLoss(nn.Module):
    def __init__(self, margin=0):
        super(CosineContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        cos_sim = F.cosine_similarity(output1, output2)
        loss_cos_con = torch.mean((1-label) * torch.div(torch.pow((1.0-cos_sim), 2), 4) +
                                    (label) * torch.pow(cos_sim * torch.lt(cos_sim, self.margin), 2))
        return loss_cos_con
criterion = CosineContrastiveLoss()
optimizer = optim.Adam(model.parameters(),lr = 0.0005)


def train(epoch):
    model.train()
    for _, data in enumerate(training_loader, 0):
        ids, mask, token_type_ids = data['ids'], data['mask'], data['token_type_ids']
        targets = data['targets'].to(DEVICE, dtype=torch.float)
        ids = [ids[0].to(DEVICE, dtype=torch.long), ids[1].to(DEVICE, dtype=torch.long)]
        mask = [mask[0].to(DEVICE, dtype=torch.long), mask[1].to(DEVICE, dtype=torch.long)]
        token_type_ids = [token_type_ids[0].to(DEVICE, dtype=torch.long),
                          token_type_ids[1].to(DEVICE, dtype=torch.long)]
        output1, output2 = model(ids, mask, token_type_ids)
        optimizer.zero_grad()
        loss = criterion(torch.tensor(output1), torch.tensor(output2), torch.tensor(targets))
        if _ % 5000 == 0:
            print(f'Step: {_}, Epoch: {epoch}, Loss:  {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
for epoch in range(EPOCHS):
    train(epoch)
def validation():
    model.eval()
    fin_targets=[]
    fin_outputs=[]
    with torch.no_grad():
        for _, data in enumerate(testing_loader, 0):
            ids,mask,token_type_ids = data['ids'],data['mask'],data['token_type_ids']
            targets = data['targets'].to(DEVICE, dtype = torch.float)
            ids = [ids[0].to(DEVICE, dtype = torch.long),ids[1].to(DEVICE, dtype = torch.long)]
            mask = [mask[0].to(DEVICE, dtype = torch.long),mask[1].to(DEVICE, dtype = torch.long)]
            token_type_ids = [token_type_ids[0].to(DEVICE, dtype = torch.long),token_type_ids[1].to(DEVICE, dtype = torch.long)]
            targets = data['targets'].to(DEVICE, dtype = torch.float)
            output1,output2 = model(ids, mask, token_type_ids)
            cos_sim = F.cosine_similarity(output1, output2)
            fin_targets.extend(targets.cpu().detach().numpy().tolist())
            fin_outputs.extend(torch.sigmoid(cos_sim).cpu().detach().numpy().tolist())
    return fin_outputs, fin_targets
outputs, targets = validation()
outputs = np.array(outputs) >= 0.5
accuracy = metrics.accuracy_score(targets, outputs)
f1_score_micro = metrics.f1_score(targets, outputs, average='micro')
f1_score_macro = metrics.f1_score(targets, outputs, average='macro')
print(f"Accuracy Score = {accuracy}")
print(f"F1 Score (Micro) = {f1_score_micro}")
print(f"F1 Score (Macro) = {f1_score_macro}")
